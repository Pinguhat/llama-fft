"""
Unified correctness + performance testbench for llama-fft.

What it does in ONE run:
- Loads a teacher (original) model once, caches teacher last-token logits on CPU, then frees teacher
- For each block size B:
    - loads a fresh student model
    - patches first N layers with BlockCirculantLinear (repo patch)
    - optionally loads calibrated BC params (save_bc_params/load_bc_params format)
    - correctness: MSE/KL/cosine + top1 + topk-overlap on last token
    - performance: prefill forward time + tokens/s; optional generate decode tokens/s; peak VRAM
- Writes JSON + CSV

Notes:
- Optional caching of FFT(c): enabled only for performance suite by default.
  This does NOT change the math when weights are fixed; it only avoids recomputing rFFT(c) every forward.
"""

import argparse
import csv
import gc
import inspect
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

import patch_llama_fft as plf


# -----------------------------
# Small utilities
# -----------------------------

def pick_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32

def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()

def read_prompts(path: str, limit: int) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
            if len(out) >= limit:
                break
    if not out:
        raise RuntimeError(f"No prompts loaded from {path}")
    return out

def tokenize_prompts(
    tokenizer,
    prompts: List[str],
    max_len: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns list of (input_ids_1d, attention_mask_1d) on CPU.
    """
    items: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for text in prompts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        items.append((enc["input_ids"][0].cpu(), enc["attention_mask"][0].cpu()))
    return items

def pad_batch(items: List[Tuple[torch.Tensor, torch.Tensor]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    items: list of (input_ids_1d, attention_mask_1d), CPU
    returns:
      input_ids (B,T), attention_mask (B,T), last_idx (B,)
    """
    bsz = len(items)
    max_len = max(int(x[0].numel()) for x in items)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    last_idx = torch.zeros((bsz,), dtype=torch.long)

    for i, (ids, attn) in enumerate(items):
        n = int(ids.numel())
        input_ids[i, :n] = ids
        attention_mask[i, :n] = attn
        last_idx[i] = int(attn.sum().item()) - 1
    return input_ids, attention_mask, last_idx

def iter_batches(tokenized: List[Tuple[torch.Tensor, torch.Tensor]], batch_size: int, pad_id: int):
    for i in range(0, len(tokenized), batch_size):
        batch = tokenized[i:i+batch_size]
        yield pad_batch(batch, pad_id)

def gather_last_logits(logits: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
    """
    logits: (B,T,V), last_idx: (B,)
    returns (B,V)
    """
    bsz = logits.shape[0]
    idx = last_idx.view(bsz, 1, 1).expand(bsz, 1, logits.shape[-1])
    return logits.gather(dim=1, index=idx).squeeze(1)

def softmax_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    KL(teacher || student) over vocab dim; both logits float32; returns shape (B,)
    """
    t = float(temperature)
    p = torch.softmax(teacher_logits / t, dim=-1)
    log_q = torch.log_softmax(student_logits / t, dim=-1)
    log_p = torch.log_softmax(teacher_logits / t, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


# -----------------------------
# Patching helpers (robust to signature changes)
# -----------------------------

def patch_model_with_repo_patch(model, block_size: int, num_layers: int) -> None:
    fn = plf.patch_mlp_with_block_circulant
    sig = inspect.signature(fn)
    kwargs = {}
    if "block_size" in sig.parameters:
        kwargs["block_size"] = block_size
    if "num_layers_to_patch" in sig.parameters:
        kwargs["num_layers_to_patch"] = num_layers
    if "num_layers" in sig.parameters:  # just in case
        kwargs["num_layers"] = num_layers
    fn(model, **kwargs) if kwargs else fn(model)

def maybe_load_calib(model, calib_path: Optional[str]) -> Tuple[int, int, int]:
    if not calib_path:
        return (0, 0, 0)
    if not os.path.exists(calib_path):
        return (0, 0, 0)

    if hasattr(plf, "load_bc_params") and callable(getattr(plf, "load_bc_params")):
        try:
            plf.load_bc_params(model, calib_path)
            return (1, 0, 0)
        except Exception:
            pass

    ckpt = torch.load(calib_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    return (1, len(missing), len(unexpected))


# -----------------------------
# Optional: cache FFT(c) inside BlockCirculantLinear for perf runs
# (safe for inference; DO NOT use during calibration/training)
# -----------------------------

def enable_cfft_cache(model) -> int:
    """
    Monkey-patch BlockCirculantLinear.forward to reuse cached rFFT(c).
    Returns number of patched BC layers.
    """
    if not hasattr(plf, "BlockCirculantLinear"):
        return 0

    BC = plf.BlockCirculantLinear
    patched = 0

    for m in model.modules():
        if isinstance(m, BC) and not hasattr(m, "_orig_forward"):
            m._orig_forward = m.forward  # type: ignore[attr-defined]

            def forward_cached(self, x: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
                # Same logic as your vectorized forward, but uses cached Cf when available.
                if x.dim() == 3:
                    bs, sl, in_f = x.shape
                    if in_f != self.in_features:
                        raise ValueError("in_features mismatch")
                    x_flat = x.reshape(bs * sl, in_f)
                    is_3d = True
                elif x.dim() == 2:
                    x_flat = x
                    bs = x.shape[0]
                    sl = 1
                    is_3d = False
                else:
                    raise ValueError("Unsupported input dim")

                dtype = x_flat.dtype
                B = self.block_size

                # safer than .view() if non-contiguous
                x_blocks = x_flat.reshape(x_flat.shape[0], self.in_blocks, B)

                x32 = x_blocks.to(torch.float32)
                Xf = torch.fft.rfft(x32, dim=-1)

                # Cache Cf per (device, B, shape) in inference mode
                dev = x_flat.device
                key = (str(dev), int(B), int(self.out_blocks), int(self.in_blocks))
                cache = getattr(self, "_Cf_cache", None)
                if cache is None:
                    cache = {}
                    setattr(self, "_Cf_cache", cache)

                if (not self.training) and (not torch.is_grad_enabled()):
                    Cf = cache.get(key, None)
                    if Cf is None:
                        Cf = torch.fft.rfft(self.c.to(device=dev, dtype=torch.float32), dim=-1)
                        cache[key] = Cf
                else:
                    # training/calibration: never reuse cached Cf
                    Cf = torch.fft.rfft(self.c.to(device=dev, dtype=torch.float32), dim=-1)

                Yf = torch.einsum("oif,nif->nof", Cf, Xf)
                y32 = torch.fft.irfft(Yf, n=B, dim=-1)
                y_blocks = y32.to(dtype)

                y_flat = y_blocks.reshape(x_flat.shape[0], self.out_features)
                if self.bias is not None:
                    y_flat = y_flat + self.bias

                if is_3d:
                    return y_flat.reshape(bs, sl, self.out_features)
                return y_flat

            # bind method
            m.forward = forward_cached.__get__(m, m.__class__)  # type: ignore[method-assign]
            patched += 1

    return patched

def disable_cfft_cache(model) -> int:
    """
    Restore original forward if we monkey patched it.
    """
    if not hasattr(plf, "BlockCirculantLinear"):
        return 0
    BC = plf.BlockCirculantLinear
    restored = 0
    for m in model.modules():
        if isinstance(m, BC) and hasattr(m, "_orig_forward"):
            m.forward = m._orig_forward  # type: ignore[method-assign]
            delattr(m, "_orig_forward")
            if hasattr(m, "_Cf_cache"):
                delattr(m, "_Cf_cache")
            restored += 1
    return restored


# -----------------------------
# Suites
# -----------------------------

@torch.inference_mode()
def compute_teacher_last_cache(
    model_path: str,
    tokenizer,
    tokenized: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """
    Returns list of teacher last-token logits on CPU float32: [(V,), ...]
    """
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map=None,
    ).to(device)
    teacher.eval()

    out: List[torch.Tensor] = []
    for ids, attn in tokenized:
        input_ids = ids.unsqueeze(0).to(device)
        attention_mask = attn.unsqueeze(0).to(device)
        logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        last_idx = int(attn.sum().item()) - 1
        out.append(logits[0, last_idx, :].to(torch.float32).cpu())

    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


@torch.inference_mode()
def correctness_last_token(
    student,
    tokenized: List[Tuple[torch.Tensor, torch.Tensor]],
    teacher_last_cache: List[torch.Tensor],
    device: torch.device,
    temperature: float,
    topk: int,
) -> Dict[str, float]:
    mse_sum = 0.0
    kl_sum = 0.0
    cos_sum = 0.0
    top1 = 0
    topk_in = 0
    topk_overlap_sum = 0.0
    n = 0

    for (ids, attn), t_last_cpu in zip(tokenized, teacher_last_cache):
        input_ids = ids.unsqueeze(0).to(device)
        attention_mask = attn.unsqueeze(0).to(device)

        s_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
        s_last = s_logits[0, int(attn.sum().item()) - 1, :].to(torch.float32)
        t_last = t_last_cpu.to(device)

        mse_sum += (s_last - t_last).pow(2).mean().item()
        kl_sum += softmax_kl(s_last, t_last, temperature=temperature).mean().item()
        cos_sum += F.cosine_similarity(s_last.unsqueeze(0), t_last.unsqueeze(0), dim=-1).mean().item()

        t_top1 = int(t_last.argmax(dim=-1).item())
        s_top1 = int(s_last.argmax(dim=-1).item())
        top1 += int(t_top1 == s_top1)

        s_topk = torch.topk(s_last, k=topk, dim=-1).indices.tolist()
        t_topk = torch.topk(t_last, k=topk, dim=-1).indices.tolist()

        topk_in += int(t_top1 in s_topk)
        topk_overlap_sum += float(len(set(s_topk).intersection(set(t_topk))) / float(topk))

        n += 1

    denom = float(max(n, 1))
    return {
        "last_mse": mse_sum / denom,
        "last_kl": kl_sum / denom,
        "last_cos": cos_sum / denom,
        "last_top1_acc": float(top1) / denom,
        "last_top1_in_student_topk": float(topk_in) / denom,
        "last_topk_overlap": topk_overlap_sum / denom,
    }


@torch.inference_mode()
def perf_prefill(
    model,
    tokenized: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    pad_id: int,
    batch_size: int,
    warmup: int,
    runs: int,
) -> Dict[str, float]:
    batches = list(iter_batches(tokenized, batch_size, pad_id))
    # move once per batch
    gpu_batches = []
    token_counts = []
    for input_ids, attention_mask, _last_idx in batches:
        token_counts.append(int(attention_mask.sum().item()))
        gpu_batches.append((input_ids.to(device), attention_mask.to(device)))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        for input_ids, attention_mask in gpu_batches:
            _ = model(input_ids=input_ids, attention_mask=attention_mask).logits
    sync_if_cuda(device)

    t0 = time.perf_counter()
    for _ in range(runs):
        for input_ids, attention_mask in gpu_batches:
            _ = model(input_ids=input_ids, attention_mask=attention_mask).logits
    sync_if_cuda(device)
    t1 = time.perf_counter()

    total_forwards = runs * len(gpu_batches)
    total_tokens = runs * sum(token_counts)
    total_s = max(t1 - t0, 1e-9)

    avg_ms = (total_s / max(total_forwards, 1)) * 1000.0
    tok_s = total_tokens / total_s
    peak = float(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0.0

    return {"prefill_avg_ms": avg_ms, "prefill_tokens_per_s": tok_s, "peak_mem_bytes": peak}


@torch.inference_mode()
def perf_generate(
    model,
    tokenized: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    pad_id: int,
    batch_size: int,
    warmup: int,
    runs: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    batches = list(iter_batches(tokenized, batch_size, pad_id))
    gpu_batches = []
    for input_ids, attention_mask, _last_idx in batches:
        gpu_batches.append((input_ids.to(device), attention_mask.to(device)))

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)

    def do_gen():
        for input_ids, attention_mask in gpu_batches:
            _ = model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)

    for _ in range(warmup):
        do_gen()
    sync_if_cuda(device)

    t0 = time.perf_counter()
    for _ in range(runs):
        do_gen()
    sync_if_cuda(device)
    t1 = time.perf_counter()

    total_s = max(t1 - t0, 1e-9)
    total_new = runs * len(tokenized) * max_new_tokens
    new_tok_s = total_new / total_s
    return {"decode_new_tokens_per_s": new_tok_s, "decode_total_s": total_s}


# -----------------------------
# Main
# -----------------------------

@dataclass
class Row:
    B: int
    calib_loaded: int
    missing_keys: int
    unexpected_keys: int

    # correctness
    last_mse: float
    last_kl: float
    last_cos: float
    last_top1_acc: float
    last_top1_in_student_topk: float
    last_topk_overlap: float

    # perf
    prefill_avg_ms: float
    prefill_tokens_per_s: float
    peak_mem_bytes: float
    decode_new_tokens_per_s: float

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompts_file", type=str, required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=128)

    ap.add_argument("--block_sizes", type=str, default="64,128,256")
    ap.add_argument("--num_layers", type=int, default=1)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)

    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=5)

    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--no_generate", action="store_true")

    ap.add_argument("--calib_dir", type=str, default="")
    ap.add_argument("--csv_out", type=str, default="bench_all.csv")
    ap.add_argument("--json_out", type=str, default="bench_all.json")

    # caching control
    ap.add_argument("--cache_cfft", type=int, default=1, choices=[0, 1])
    ap.add_argument("--cache_for_correctness", type=int, default=0, choices=[0, 1])

    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype)
    prompts = read_prompts(args.prompts_file, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        # typical for llama
        tokenizer.pad_token = tokenizer.eos_token

    pad_id = int(tokenizer.pad_token_id)

    # Tokenize once
    tokenized = tokenize_prompts(tokenizer, prompts, args.max_len)

    # Teacher cache once (last-token only), then free teacher
    print("Computing teacher last-token cache (once)...")
    teacher_last_cache = compute_teacher_last_cache(
        model_path=args.model_path,
        tokenizer=tokenizer,
        tokenized=tokenized,
        device=device,
        dtype=dtype,
    )

    block_sizes = [int(x.strip()) for x in args.block_sizes.split(",") if x.strip()]
    if not block_sizes:
        raise ValueError("No block sizes specified")

    rows: List[Row] = []
    meta: Dict[str, Any] = {
        "model_path": args.model_path,
        "device": str(device),
        "dtype": args.dtype,
        "num_layers": args.num_layers,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "runs": args.runs,
        "warmup": args.warmup,
        "max_new_tokens": args.max_new_tokens,
        "cache_cfft": args.cache_cfft,
        "cache_for_correctness": args.cache_for_correctness,
        "no_generate": bool(args.no_generate),
    }

    for B in block_sizes:
        print(f"\n=== B={B} ===")

        student = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map=None,
        ).to(device)
        student.eval()

        patch_model_with_repo_patch(student, block_size=B, num_layers=args.num_layers)

        # Optional: load calibrated BC params
        calib_path = None
        if args.calib_dir:
            cand = os.path.join(args.calib_dir, f"bc_calibrated_B{B}.pt")
            if os.path.exists(cand):
                calib_path = cand

        loaded, missing_ct, unexpected_ct = maybe_load_calib(student, calib_path)

        # caching policy
        if args.cache_cfft and (not args.cache_for_correctness):
            nbc = enable_cfft_cache(student)
            print(f"Enabled cFFT cache for perf: patched {nbc} BC layers")

        # correctness (optionally with cache; default off)
        if (not args.cache_for_correctness) and args.cache_cfft:
            # ensure correctness uses non-cached path unless explicitly requested
            _ = disable_cfft_cache(student)

        corr = correctness_last_token(
            student=student,
            tokenized=tokenized,
            teacher_last_cache=teacher_last_cache,
            device=device,
            temperature=args.temperature,
            topk=args.topk,
        )

        # perf: enable cache for perf if requested
        if args.cache_cfft:
            _ = enable_cfft_cache(student)

        perf = perf_prefill(
            model=student,
            tokenized=tokenized,
            device=device,
            pad_id=pad_id,
            batch_size=args.batch_size,
            warmup=args.warmup,
            runs=args.runs,
        )

        decode = {"decode_new_tokens_per_s": 0.0}
        if not args.no_generate:
            decode = perf_generate(
                model=student,
                tokenized=tokenized,
                device=device,
                pad_id=pad_id,
                batch_size=args.batch_size,
                warmup=max(0, args.warmup - 1),
                runs=max(1, args.runs),
                max_new_tokens=args.max_new_tokens,
            )

        row = Row(
            B=B,
            calib_loaded=int(loaded),
            missing_keys=int(missing_ct),
            unexpected_keys=int(unexpected_ct),

            last_mse=float(corr["last_mse"]),
            last_kl=float(corr["last_kl"]),
            last_cos=float(corr["last_cos"]),
            last_top1_acc=float(corr["last_top1_acc"]),
            last_top1_in_student_topk=float(corr["last_top1_in_student_topk"]),
            last_topk_overlap=float(corr["last_topk_overlap"]),

            prefill_avg_ms=float(perf["prefill_avg_ms"]),
            prefill_tokens_per_s=float(perf["prefill_tokens_per_s"]),
            peak_mem_bytes=float(perf["peak_mem_bytes"]),
            decode_new_tokens_per_s=float(decode.get("decode_new_tokens_per_s", 0.0)),
        )
        rows.append(row)
        print(asdict(row))

        del student
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # write outputs
    out_json = {"meta": meta, "rows": [asdict(r) for r in rows]}
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    print("\nWrote:", args.csv_out)
    print("Wrote:", args.json_out)


if __name__ == "__main__":
    main()
