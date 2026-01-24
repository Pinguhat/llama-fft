#!/usr/bin/env python3
import os
import time
import csv
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Block-circulant layer + patch
# -----------------------------

def circulant_matvec_fft(c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # C(c)_{i,j} = c[(i-j) mod B]
    # Compute y = C x via FFT: y = irfft(rfft(c) * rfft(x))
    assert c.dim() == 1 and x.dim() == 1
    n = c.shape[0]
    assert x.shape[0] == n

    orig_dtype = x.dtype
    c32 = c.to(torch.float32)
    x32 = x.to(torch.float32)

    fft_c = torch.fft.rfft(c32)
    fft_x = torch.fft.rfft(x32)
    fft_y = fft_c * fft_x
    y32 = torch.fft.irfft(fft_y, n=n)

    return y32.to(orig_dtype)


def dense_block_to_circulant_column_loss_aware(W_block: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Frobenius-optimal projection onto circulant + optimal scalar alpha.
    Convention: C[i,j] = c[(i-j) mod B], where c is FIRST column.
    """
    assert W_block.dim() == 2
    B0, B1 = W_block.shape
    assert B0 == B1
    B = B0

    device = W_block.device
    dtype = W_block.dtype
    idx = torch.arange(B, device=device)

    c = torch.empty(B, device=device, dtype=dtype)
    diag_sums = torch.empty(B, device=device, dtype=dtype)

    for t in range(B):
        cols = (idx - t) % B
        vals = W_block[idx, cols]
        c[t] = vals.mean()
        diag_sums[t] = vals.sum()

    # alpha = <W, C> / <C, C>
    numerator = (c * diag_sums).sum()
    denom = (B * (c * c).sum()).clamp_min(eps)
    alpha = numerator / denom

    return (alpha * c).to(dtype)


class BlockCirculantLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, block_size: int, bias: bool = True):
        super().__init__()
        assert in_features % block_size == 0
        assert out_features % block_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.in_blocks = in_features // block_size
        self.out_blocks = out_features // block_size

        self.c = nn.Parameter(torch.randn(self.out_blocks, self.in_blocks, self.block_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int) -> "BlockCirculantLinear":
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            block_size=block_size,
            bias=(linear.bias is not None),
        )

        with torch.no_grad():
            W = linear.weight.data.clone()  # (out_f, in_f)
            B = block_size
            out_blocks = layer.out_blocks
            in_blocks = layer.in_blocks

            # (out_blocks, B, in_blocks, B) -> (out_blocks, in_blocks, B, B)
            W_blocks = W.view(out_blocks, B, in_blocks, B).permute(0, 2, 1, 3)

            for j in range(out_blocks):
                for i in range(in_blocks):
                    block = W_blocks[j, i]
                    c_ji = dense_block_to_circulant_column_loss_aware(block)
                    layer.c.data[j, i].copy_(c_ji)

            if layer.bias is not None and linear.bias is not None:
                layer.bias.data.copy_(linear.bias.data)

        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Supports (batch, in_features) and (batch, seq, in_features)
        if x.dim() == 3:
            bs, sl, in_f = x.shape
            assert in_f == self.in_features
            x_flat = x.reshape(bs * sl, in_f)
            is_3d = True
        elif x.dim() == 2:
            x_flat = x
            bs = x.shape[0]
            sl = 1
            is_3d = False
        else:
            raise ValueError("Unsupported input dim")

        device = x_flat.device
        dtype = x_flat.dtype

        x_blocks = x_flat.view(x_flat.shape[0], self.in_blocks, self.block_size)
        y_blocks = torch.zeros(x_flat.shape[0], self.out_blocks, self.block_size, device=device, dtype=dtype)

        # NOTE: This is not optimized. For performance benchmarking of FFT,
        # you should vectorize and cache FFT(c). This script measures end-to-end as-is.
        for b in range(x_flat.shape[0]):
            for j in range(self.out_blocks):
                acc = torch.zeros(self.block_size, device=device, dtype=dtype)
                for i in range(self.in_blocks):
                    acc = acc + circulant_matvec_fft(self.c[j, i], x_blocks[b, i])
                y_blocks[b, j] = acc

        y_flat = y_blocks.view(x_flat.shape[0], self.out_features)
        if self.bias is not None:
            y_flat = y_flat + self.bias

        if is_3d:
            return y_flat.view(bs, sl, self.out_features)
        return y_flat


def patch_mlp_layers(model: AutoModelForCausalLM, block_size: int, num_layers: int = 1) -> None:
    n_layers = model.config.num_hidden_layers
    num_layers = min(num_layers, n_layers)

    for layer_idx in range(num_layers):
        mlp = model.model.layers[layer_idx].mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            old = getattr(mlp, name)
            if not isinstance(old, nn.Linear):
                continue
            new = BlockCirculantLinear.from_linear(old, block_size=block_size)
            setattr(mlp, name, new)


def maybe_load_calib(model: AutoModelForCausalLM, calib_path: Optional[str]) -> Tuple[int, int]:
    """
    Try to load a calibration checkpoint as a state_dict.
    Returns (missing_count, unexpected_count).
    """
    if calib_path is None:
        return (0, 0)
    if not os.path.exists(calib_path):
        return (0, 0)

    ckpt = torch.load(calib_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        return (0, 0)

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    return (len(missing), len(unexpected))


# -----------------------------
# Metrics + benchmarking
# -----------------------------

@dataclass
class Metrics:
    logit_mse: float
    logit_kl: float
    token_agree: float


def softmax_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # KL(teacher || student) over last dimension
    t = temperature
    p = torch.softmax(teacher_logits / t, dim=-1)
    log_q = torch.log_softmax(student_logits / t, dim=-1)
    log_p = torch.log_softmax(teacher_logits / t, dim=-1)
    kl = (p * (log_p - log_q)).sum(dim=-1)  # (..., seq)
    return kl


def read_prompts(path: str, limit: int) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            prompts.append(s)
            if len(prompts) >= limit:
                break
    return prompts


@torch.no_grad()
def eval_one_model_pair(
    teacher: AutoModelForCausalLM,
    student: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    max_len: int,
    temperature: float,
) -> Metrics:
    mse_sum = 0.0
    kl_sum = 0.0
    agree_sum = 0.0
    count = 0

    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        out_t = teacher(**inputs).logits
        out_s = student(**inputs).logits

        # Align shapes (batch, seq, vocab)
        assert out_t.shape == out_s.shape

        # MSE on logits
        mse = (out_s - out_t).float().pow(2).mean().item()

        # KL on logits
        kl = softmax_kl(out_s.float(), out_t.float(), temperature=temperature).mean().item()

        # Token agreement (argmax per position)
        tok_t = out_t.argmax(dim=-1)
        tok_s = out_s.argmax(dim=-1)
        agree = (tok_t == tok_s).float().mean().item()

        mse_sum += mse
        kl_sum += kl
        agree_sum += agree
        count += 1

    return Metrics(
        logit_mse=mse_sum / max(count, 1),
        logit_kl=kl_sum / max(count, 1),
        token_agree=agree_sum / max(count, 1),
    )


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def measure_forward_time_ms(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    device: torch.device,
    max_len: int,
    warmup: int,
    runs: int,
) -> Tuple[float, float]:
    # Returns (avg_ms_per_forward, tokens_per_s) using input token count
    token_counts = []
    input_batches = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        token_counts.append(int(inputs["input_ids"].numel()))
        input_batches.append({k: v.to(device) for k, v in inputs.items()})

    # Warmup
    for _ in range(warmup):
        for inp in input_batches:
            _ = model(**inp).logits
    sync_if_cuda(device)

    # Timed runs
    t0 = time.perf_counter()
    for _ in range(runs):
        for inp in input_batches:
            _ = model(**inp).logits
    sync_if_cuda(device)
    t1 = time.perf_counter()

    total_forwards = runs * len(input_batches)
    total_tokens = runs * sum(token_counts)
    total_s = max(t1 - t0, 1e-9)

    avg_ms = (total_s / max(total_forwards, 1)) * 1000.0
    tok_s = total_tokens / total_s
    return avg_ms, tok_s


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--prompts_file", type=str, required=True)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--block_sizes", type=str, default="64,128,256")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--calib_dir", type=str, default="")
    ap.add_argument("--csv_out", type=str, default="bench_results.csv")
    ap.add_argument("--plot_out", type=str, default="bench_plot.png")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    block_sizes = [int(x.strip()) for x in args.block_sizes.split(",") if x.strip()]

    prompts = read_prompts(args.prompts_file, args.limit)
    if len(prompts) == 0:
        raise RuntimeError("No prompts loaded")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    # Teacher (original)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        device_map=None,
    ).to(device)
    teacher.eval()

    rows: List[Dict[str, float]] = []

    for B in block_sizes:
        # Student (patched)
        student = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map=None,
        ).to(device)
        student.eval()

        patch_mlp_layers(student, block_size=B, num_layers=args.num_layers)

        calib_path = None
        if args.calib_dir:
            # Convention: bc_calibrated_B64.pt etc (adjust if you use a different naming)
            cand = os.path.join(args.calib_dir, f"bc_calibrated_B{B}.pt")
            if os.path.exists(cand):
                calib_path = cand

        missing_ct, unexpected_ct = maybe_load_calib(student, calib_path)

        # Accuracy / similarity metrics
        metrics = eval_one_model_pair(
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            prompts=prompts,
            device=device,
            max_len=args.max_len,
            temperature=args.temperature,
        )

        # Performance metrics
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
            "logit_mse": metrics.logit_mse,
            "logit_kl": metrics.logit_kl,
            "token_agree": metrics.token_agree,
            "avg_ms": avg_ms,
            "tokens_per_s": tok_s,
            "calib_loaded": 1.0 if calib_path else 0.0,
            "missing_keys": float(missing_ct),
            "unexpected_keys": float(unexpected_ct),
        }
        rows.append(row)

        # Free VRAM between runs
        del student
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Write CSV
    with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Plot (single figure, dual axis)
    Bs = [int(r["B"]) for r in rows]
    tokens_per_s = [r["tokens_per_s"] for r in rows]
    # Choose one accuracy metric for the main plot (KL is often easiest to explain: smaller is better)
    logit_kl = [r["logit_kl"] for r in rows]

    fig, ax1 = plt.subplots()
    ax1.plot(Bs, logit_kl, marker="o")
    ax1.set_xlabel("Block size B")
    ax1.set_ylabel("Logit KL (teacher || student), lower is better")

    ax2 = ax1.twinx()
    ax2.plot(Bs, tokens_per_s, marker="s")
    ax2.set_ylabel("Tokens per second, higher is better")

    ax1.set_xticks(Bs)
    fig.tight_layout()
    fig.savefig(args.plot_out, dpi=200)

    print("Wrote:", args.csv_out)
    print("Wrote:", args.plot_out)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
