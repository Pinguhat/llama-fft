#!/usr/bin/env python3
import os
import time
import csv
import argparse
import gc
import inspect
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

import patch_llama_fft as plf


# -----------------------------
# Metrics
# -----------------------------

@dataclass
class Metrics:
    logit_mse: float
    logit_kl: float
    token_agree: float


def softmax_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # KL(teacher || student) over vocab dim, returns shape (..., seq)
    t = float(temperature)
    p = torch.softmax(teacher_logits / t, dim=-1)
    log_q = torch.log_softmax(student_logits / t, dim=-1)
    log_p = torch.log_softmax(teacher_logits / t, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


def read_prompts(path: str, limit: int) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
            if len(out) >= limit:
                break
    return out


def pick_dtype(s: str) -> torch.dtype:
    if s == "float16":
        return torch.float16
    if s == "bfloat16":
        return torch.bfloat16
    return torch.float32


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def patch_model_with_repo_patch(model, block_size: int, num_layers: int) -> None:
    """
    Call patch_llama_fft.patch_mlp_with_block_circulant with a signature that may differ across versions.
    """
    fn = plf.patch_mlp_with_block_circulant
    sig = inspect.signature(fn)
    kwargs = {}
    # Common variants:
    # patch_mlp_with_block_circulant(model)
    # patch_mlp_with_block_circulant(model, block_size=..., num_layers_to_patch=...)
    # patch_mlp_with_block_circulant(model, num_layers_to_patch=..., block_size=...)
    if "block_size" in sig.parameters:
        kwargs["block_size"] = block_size
    if "num_layers_to_patch" in sig.parameters:
        kwargs["num_layers_to_patch"] = num_layers
    fn(model, **kwargs) if kwargs else fn(model)


def maybe_load_calib(model, calib_path: Optional[str]) -> Tuple[int, int, int]:
    """
    Try to load calibration params.
    Returns (loaded_flag, missing_count, unexpected_count).
    """
    if not calib_path:
        return (0, 0, 0)
    if not os.path.exists(calib_path):
        return (0, 0, 0)

    # Prefer repo helper if present
    if hasattr(plf, "load_bc_params") and callable(getattr(plf, "load_bc_params")):
        try:
            plf.load_bc_params(model, calib_path)
            return (1, 0, 0)
        except Exception:
            pass

    # Fallback: load state dict
    ckpt = torch.load(calib_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    return (1, len(missing), len(unexpected))


@torch.inference_mode()
def compute_teacher_cache(
    model_path: str,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    dtype: torch.dtype,
    max_len: int,
) -> List[torch.Tensor]:
    """
    Compute teacher logits once, store on CPU (float16 to save RAM).
    """
    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map=None,
    ).to(device)
    teacher.eval()

    cache: List[torch.Tensor] = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = teacher(**inputs).logits  # (1, seq, vocab)
        cache.append(logits.detach().to("cpu"))
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


@torch.inference_mode()
def eval_student_vs_teacher_cache(
    student,
    tokenizer,
    prompts: List[str],
    teacher_cache: List[torch.Tensor],
    device: torch.device,
    max_len: int,
    temperature: float,
) -> Metrics:
    mse_sum = 0.0
    kl_sum = 0.0
    agree_sum = 0.0
    n = 0

    for text, t_logits_cpu in zip(prompts, teacher_cache):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        s_logits = student(**inputs).logits  # (1, seq, vocab)
        t_logits = t_logits_cpu.to(device)

        # MSE on logits
        mse = (s_logits.float() - t_logits.float()).pow(2).mean().item()

        # KL on logits
        kl = softmax_kl(s_logits.float(), t_logits.float(), temperature=temperature).mean().item()

        # Token agreement
        tok_t = t_logits.argmax(dim=-1)
        tok_s = s_logits.argmax(dim=-1)
        agree = (tok_t == tok_s).float().mean().item()

        mse_sum += mse
        kl_sum += kl
        agree_sum += agree
        n += 1

    return Metrics(
        logit_mse=mse_sum / max(n, 1),
        logit_kl=kl_sum / max(n, 1),
        token_agree=agree_sum / max(n, 1),
    )


@torch.inference_mode()
def measure_forward_time_ms(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_len: int,
    warmup: int,
    runs: int,
) -> Tuple[float, float]:
    token_counts = []
    batches = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        token_counts.append(int(inputs["input_ids"].numel()))
        batches.append({k: v.to(device) for k, v in inputs.items()})

    for _ in range(warmup):
        for inp in batches:
            _ = model(**inp).logits
    sync_if_cuda(device)

    t0 = time.perf_counter()
    for _ in range(runs):
        for inp in batches:
            _ = model(**inp).logits
    sync_if_cuda(device)
    t1 = time.perf_counter()

    total_forwards = runs * len(batches)
    total_tokens = runs * sum(token_counts)
    total_s = max(t1 - t0, 1e-9)

    avg_ms = (total_s / max(total_forwards, 1)) * 1000.0
    tok_s = total_tokens / total_s
    return avg_ms, tok_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompts_file", type=str, required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--block_sizes", type=str, default="64,128,256")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--calib_dir", type=str, default="")
    ap.add_argument("--csv_out", type=str, default="bench_results.csv")
    ap.add_argument("--plot_out", type=str, default="bench_plot.png")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = pick_dtype(args.dtype)
    block_sizes = [int(x.strip()) for x in args.block_sizes.split(",") if x.strip()]

    prompts = read_prompts(args.prompts_file, args.limit)
    if not prompts:
        raise RuntimeError("No prompts loaded")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    # Some llama tokenizers have no pad token; set it for safety
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Teacher cache once (then free teacher to avoid OOM)
    print("Computing teacher cache (once)...")
    teacher_cache = compute_teacher_cache(
        model_path=args.model_path,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        dtype=dtype,
        max_len=args.max_len,
    )

    rows: List[Dict[str, float]] = []

    for B in block_sizes:
        print(f"\n=== Bench B={B} ===")

        student = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map=None,
        ).to(device)
        student.eval()

        # Patch using repo implementation (single source of truth)
        patch_model_with_repo_patch(student, block_size=B, num_layers=args.num_layers)

        calib_path = None
        if args.calib_dir:
            cand = os.path.join(args.calib_dir, f"bc_calibrated_B{B}.pt")
            if os.path.exists(cand):
                calib_path = cand

        loaded, missing_ct, unexpected_ct = maybe_load_calib(student, calib_path)

        # Similarity metrics (student vs cached teacher)
        metrics = eval_student_vs_teacher_cache(
            student=student,
            tokenizer=tokenizer,
            prompts=prompts,
            teacher_cache=teacher_cache,
            device=device,
            max_len=args.max_len,
            temperature=args.temperature,
        )

        # Performance metrics (student only)
        avg_ms, tok_s = measure_forward_time_ms(
            model=student,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_len=args.max_len,
            warmup=args.warmup,
            runs=args.runs,
        )

        row = {
            "B": float(B),
            "logit_mse": float(metrics.logit_mse),
            "logit_kl": float(metrics.logit_kl),
            "token_agree": float(metrics.token_agree),
            "avg_ms": float(avg_ms),
            "tokens_per_s": float(tok_s),
            "calib_loaded": float(loaded),
            "missing_keys": float(missing_ct),
            "unexpected_keys": float(unexpected_ct),
        }
        rows.append(row)
        print(row)

        del student
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write CSV
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot: KL (lower is better) and tokens/s (higher is better)
    Bs = [int(r["B"]) for r in rows]
    logit_kl = [r["logit_kl"] for r in rows]
    tokens_per_s = [r["tokens_per_s"] for r in rows]

    fig, ax1 = plt.subplots()
    ax1.plot(Bs, logit_kl, marker="o")
    ax1.set_xlabel("Block size B")
    ax1.set_ylabel("Logit KL (teacher || student), lower is better")
    ax1.set_xticks(Bs)

    ax2 = ax1.twinx()
    ax2.plot(Bs, tokens_per_s, marker="s")
    ax2.set_ylabel("Tokens per second, higher is better")

    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=200)

    print("\nWrote:", args.csv_out)
    print("Wrote:", args.plot_out)


if __name__ == "__main__":
    main()
