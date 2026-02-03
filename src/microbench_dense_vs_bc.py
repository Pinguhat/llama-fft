import argparse
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Uses your implementation
from patch_llama_fft import BlockCirculantLinear


@dataclass
class BenchResult:
    name: str
    ms_per_iter: float
    iters: int


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def bench_fn(fn: Callable[[], torch.Tensor], *, device: torch.device, warmup: int, iters: int) -> float:
    # Warmup
    for _ in range(warmup):
        _ = fn()
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fn()
    _sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) * 1000.0 / float(iters)


def make_dense_linear(in_f: int, out_f: int, *, device: torch.device, dtype: torch.dtype) -> nn.Linear:
    lin = nn.Linear(in_f, out_f, bias=True).to(device=device, dtype=dtype)
    lin.eval()
    return lin


def make_bc_linear(in_f: int, out_f: int, B: int, *, device: torch.device, dtype: torch.dtype) -> BlockCirculantLinear:
    bc = BlockCirculantLinear(in_f, out_f, block_size=B, bias=True).to(device=device, dtype=dtype)
    bc.eval()
    return bc


def run_one_case(
    name: str,
    fn_dense: Callable[[], torch.Tensor],
    fn_bc: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Tuple[BenchResult, BenchResult]:
    ms_dense = bench_fn(fn_dense, device=device, warmup=warmup, iters=iters)
    ms_bc = bench_fn(fn_bc, device=device, warmup=warmup, iters=iters)
    return (
        BenchResult(name=f"{name} | dense", ms_per_iter=ms_dense, iters=iters),
        BenchResult(name=f"{name} | bc", ms_per_iter=ms_bc, iters=iters),
    )


def fmt(res: BenchResult) -> str:
    return f"{res.name:28s}: {res.ms_per_iter:9.3f} ms/iter"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="", help="float32|float16|bfloat16. Default: cpu->float32, cuda->float16")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=128)
    p.add_argument("--block_sizes", type=str, default="64,128,256")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0=leave default)")
    p.add_argument("--bench_mlp", action="store_true", help="Also benchmark LLaMA-style MLP: down(silu(gate(x))*up(x))")
    args = p.parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.threads and device.type == "cpu":
        torch.set_num_threads(args.threads)

    # dtype default
    if args.dtype.strip() == "":
        dtype = torch.float32 if device.type == "cpu" else torch.float16
    else:
        dt = args.dtype.lower().strip()
        if dt == "float32":
            dtype = torch.float32
        elif dt == "float16":
            dtype = torch.float16
        elif dt == "bfloat16":
            dtype = torch.bfloat16
        else:
            raise ValueError(f"Unknown dtype: {args.dtype}")

    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print("Note: float16/bfloat16 on CPU can be slow or emulated. float32 is usually the fairest CPU baseline.")

    Bs = [int(x.strip()) for x in args.block_sizes.split(",") if x.strip()]
    if not Bs:
        raise ValueError("No block sizes provided")

    # LLaMA 2 7B MLP shapes
    in_f = 4096
    mid_f = 11008
    out_f = 4096

    # Input tensor (3D, like transformer hidden states)
    x = torch.randn(args.batch, args.seq, in_f, device=device, dtype=dtype)

    print(f"Device: {device}, dtype: {dtype}, batch={args.batch}, seq={args.seq}, warmup={args.warmup}, iters={args.iters}")
    if device.type == "cpu":
        print(f"CPU threads: {torch.get_num_threads()}")

    # Dense modules
    dense_gate = make_dense_linear(in_f, mid_f, device=device, dtype=dtype)
    dense_up = make_dense_linear(in_f, mid_f, device=device, dtype=dtype)
    dense_down = make_dense_linear(mid_f, out_f, device=device, dtype=dtype)

    # Bench dense-only once (independent of B)
    def dense_gate_fn(): return dense_gate(x)
    def dense_up_fn(): return dense_up(x)
    def dense_down_fn():
        tmp = torch.randn(args.batch, args.seq, mid_f, device=device, dtype=dtype)
        return dense_down(tmp)

    print("\n=== Dense baselines ===")
    for fn_name, fn in [("gate 4096->11008", dense_gate_fn),
                        ("up   4096->11008", dense_up_fn),
                        ("down 11008->4096", dense_down_fn)]:
        ms = bench_fn(fn, device=device, warmup=args.warmup, iters=args.iters)
        print(f"{fn_name:20s}: {ms:9.3f} ms/iter")

    if args.bench_mlp:
        def dense_mlp_fn():
            g = dense_gate(x)
            u = dense_up(x)
            h = F.silu(g) * u
            y = dense_down(h)
            return y
        ms = bench_fn(dense_mlp_fn, device=device, warmup=args.warmup, iters=args.iters)
        print(f"{'MLP (dense)':20s}: {ms:9.3f} ms/iter")

    # BC per block size
    for B in Bs:
        print(f"\n=== Block-circulant B={B} ===")
        bc_gate = make_bc_linear(in_f, mid_f, B, device=device, dtype=dtype)
        bc_up = make_bc_linear(in_f, mid_f, B, device=device, dtype=dtype)
        bc_down = make_bc_linear(mid_f, out_f, B, device=device, dtype=dtype)

        def bc_gate_fn(): return bc_gate(x)
        def bc_up_fn(): return bc_up(x)
        def bc_down_fn():
            tmp = torch.randn(args.batch, args.seq, mid_f, device=device, dtype=dtype)
            return bc_down(tmp)

        # compare per-layer
        results = []
        d, b = run_one_case("gate 4096->11008", dense_gate_fn, bc_gate_fn, device=device, warmup=args.warmup, iters=args.iters)
        results += [d, b]
        d, b = run_one_case("up   4096->11008", dense_up_fn, bc_up_fn, device=device, warmup=args.warmup, iters=args.iters)
        results += [d, b]
        d, b = run_one_case("down 11008->4096", dense_down_fn, bc_down_fn, device=device, warmup=args.warmup, iters=args.iters)
        results += [d, b]

        for r in results:
            print(fmt(r))

        # MLP pattern
        if args.bench_mlp:
            def bc_mlp_fn():
                g = bc_gate(x)
                u = bc_up(x)
                h = F.silu(g) * u
                y = bc_down(h)
                return y

            ms_dense_mlp = bench_fn(dense_mlp_fn, device=device, warmup=args.warmup, iters=args.iters)
            ms_bc_mlp = bench_fn(bc_mlp_fn, device=device, warmup=args.warmup, iters=args.iters)
            ratio = ms_bc_mlp / ms_dense_mlp if ms_dense_mlp > 0 else float("inf")

            print(f"{'MLP (dense)':28s}: {ms_dense_mlp:9.3f} ms/iter")
            print(f"{'MLP (bc)':28s}: {ms_bc_mlp:9.3f} ms/iter")
            print(f"{'BC/Dense ratio':28s}: {ratio:9.3f}  ( <1.0 means BC faster )")


if __name__ == "__main__":
    main()
