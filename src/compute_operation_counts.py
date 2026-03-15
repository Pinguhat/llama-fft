from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


IN_FEATURES = 4096
MID_FEATURES = 11008
BLOCK_SIZES = (64, 128, 256)
PATCHED_LAYERS = (1, 8)
TOTAL_TRANSFORMER_LAYERS = 32
VOCAB_SIZE = 32000


@dataclass
class Row:
    block_size: int
    layers: int
    dense_macs: int
    dense_real_ops_mul_add: int
    fft_complex_multiplies: int
    fft_accum_complex_adds: int
    input_rfft_calls: int
    output_irfft_calls: int
    approx_fft_real_ops: int
    approx_fft_speedup_vs_dense_ops: float
    fft_complex_mul_reduction_vs_dense_macs: float
    baseline_model_linear_ops_per_token: int
    patched_model_linear_ops_per_token: int
    total_model_speedup_vs_baseline: float
    total_model_op_reduction_percent: float


def dense_macs_per_layer() -> int:
    gate = IN_FEATURES * MID_FEATURES
    up = IN_FEATURES * MID_FEATURES
    down = MID_FEATURES * IN_FEATURES
    return gate + up + down


def fft_counts_per_layer(block_size: int) -> tuple[int, int, int, int]:
    in_blocks = IN_FEATURES // block_size
    mid_blocks = MID_FEATURES // block_size
    freq_bins = block_size // 2 + 1

    # Complex multiplies from einsum over frequency bins
    # gate + up + down = 3 * (mid_blocks * in_blocks * freq_bins)
    complex_multiplies = 3 * mid_blocks * in_blocks * freq_bins

    # Complex accumulations in einsum sum over in_blocks
    # per output bin: (in_blocks - 1) complex adds
    complex_adds = 3 * mid_blocks * freq_bins * (in_blocks - 1)

    # rFFT/irFFT calls in your implementation
    # gate/up: input rFFT (in_blocks) + output iFFT (mid_blocks) each
    # down:    input rFFT (mid_blocks) + output iFFT (in_blocks)
    input_rfft_calls = 2 * in_blocks + mid_blocks
    output_irfft_calls = 2 * mid_blocks + in_blocks
    return complex_multiplies, complex_adds, input_rfft_calls, output_irfft_calls


def approx_real_ops_fft_per_layer(block_size: int) -> int:
    cmul, cadd, rfft_calls, irfft_calls = fft_counts_per_layer(block_size)
    freq_bins = block_size // 2 + 1

    # Cost model (rough, but reproducible):
    # 1 complex multiply ~= 6 real ops
    # 1 complex add      ~= 2 real ops
    # 1 real FFT call    ~= 2.5 * B * log2(B) real ops
    # 1 inverse real FFT ~= 2.5 * B * log2(B) real ops
    fft_call_cost = int(round(2.5 * block_size * (block_size.bit_length() - 1)))

    mult_ops = 6 * cmul
    add_ops = 2 * cadd
    fft_ops = (rfft_calls + irfft_calls) * fft_call_cost
    return mult_ops + add_ops + fft_ops


def build_rows() -> list[Row]:
    rows: list[Row] = []
    dense_layer = dense_macs_per_layer()
    dense_real_layer = 2 * dense_layer

    # Rough system-level baseline (linear layers only):
    # per transformer layer: attention linear (q,k,v,o) + mlp linear (gate,up,down)
    attn_linear_macs_per_layer = 4 * IN_FEATURES * IN_FEATURES
    total_linear_macs_per_layer = attn_linear_macs_per_layer + dense_layer
    lm_head_macs = IN_FEATURES * VOCAB_SIZE
    baseline_model_linear_ops_per_token = 2 * (
        TOTAL_TRANSFORMER_LAYERS * total_linear_macs_per_layer + lm_head_macs
    )

    for layers in PATCHED_LAYERS:
        for b in BLOCK_SIZES:
            cmul, cadd, rfft_calls, irfft_calls = fft_counts_per_layer(b)
            approx_fft_layer = approx_real_ops_fft_per_layer(b)

            rows.append(
                # Replace only patched MLP dense real-ops with FFT approx real-ops
                # in an otherwise unchanged model.
                Row(
                    block_size=b,
                    layers=layers,
                    dense_macs=dense_layer * layers,
                    dense_real_ops_mul_add=dense_real_layer * layers,
                    fft_complex_multiplies=cmul * layers,
                    fft_accum_complex_adds=cadd * layers,
                    input_rfft_calls=rfft_calls * layers,
                    output_irfft_calls=irfft_calls * layers,
                    approx_fft_real_ops=approx_fft_layer * layers,
                    approx_fft_speedup_vs_dense_ops=(dense_real_layer * layers) / (approx_fft_layer * layers),
                    fft_complex_mul_reduction_vs_dense_macs=(dense_layer * layers) / (cmul * layers),
                    baseline_model_linear_ops_per_token=baseline_model_linear_ops_per_token,
                    patched_model_linear_ops_per_token=(
                        baseline_model_linear_ops_per_token
                        - (dense_real_layer * layers)
                        + (approx_fft_layer * layers)
                    ),
                    total_model_speedup_vs_baseline=(
                        baseline_model_linear_ops_per_token
                        / (
                            baseline_model_linear_ops_per_token
                            - (dense_real_layer * layers)
                            + (approx_fft_layer * layers)
                        )
                    ),
                    total_model_op_reduction_percent=(
                        100.0
                        * (
                            (dense_real_layer * layers) - (approx_fft_layer * layers)
                        )
                        / baseline_model_linear_ops_per_token
                    ),
                )
            )
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "block_size",
                "patched_layers",
                "dense_macs",
                "dense_real_ops_mul_add",
                "fft_complex_multiplies",
                "fft_accum_complex_adds",
                "input_rfft_calls",
                "output_irfft_calls",
                "approx_fft_real_ops",
                "approx_fft_speedup_vs_dense_ops",
                "fft_complex_mul_reduction_vs_dense_macs",
                "baseline_model_linear_ops_per_token",
                "patched_model_linear_ops_per_token",
                "total_model_speedup_vs_baseline",
                "total_model_op_reduction_percent",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.block_size,
                    r.layers,
                    r.dense_macs,
                    r.dense_real_ops_mul_add,
                    r.fft_complex_multiplies,
                    r.fft_accum_complex_adds,
                    r.input_rfft_calls,
                    r.output_irfft_calls,
                    r.approx_fft_real_ops,
                    f"{r.approx_fft_speedup_vs_dense_ops:.4f}",
                    f"{r.fft_complex_mul_reduction_vs_dense_macs:.4f}",
                    r.baseline_model_linear_ops_per_token,
                    r.patched_model_linear_ops_per_token,
                    f"{r.total_model_speedup_vs_baseline:.4f}",
                    f"{r.total_model_op_reduction_percent:.4f}",
                ]
            )


def write_markdown(rows: list[Row], path: Path) -> None:
    lines = [
        "# Operation Counts (Per Token)",
        "",
        "Assumption: only patched MLP projections (gate, up, down) are counted.",
        "",
        "| B | Layers | Dense MACs | Dense real ops (mul+add) | FFT complex mul | FFT complex add | rFFT calls | iFFT calls | Approx FFT real ops |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r.block_size} | {r.layers} | {r.dense_macs} | {r.dense_real_ops_mul_add} | "
            f"{r.fft_complex_multiplies} | {r.fft_accum_complex_adds} | {r.input_rfft_calls} | "
            f"{r.output_irfft_calls} | {r.approx_fft_real_ops} |"
        )

    lines.extend(
        [
            "",
            "## Relative factors vs Dense",
            "",
            "- `Dense/FFT approx real-ops factor` > 1 means FFT path uses fewer estimated real operations.",
            "- `Dense MAC / FFT complex-mul factor` is a lower-level kernel ratio (ignores FFT call overhead).",
            "",
            "| B | Layers | Dense/FFT approx real-ops factor | Dense MAC / FFT complex-mul factor |",
            "|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r.block_size} | {r.layers} | {r.approx_fft_speedup_vs_dense_ops:.2f}x | "
            f"{r.fft_complex_mul_reduction_vs_dense_macs:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## System-level estimate vs full model baseline",
            "",
            "Assumptions for this section:",
            "- Llama-style decoder with 32 transformer layers.",
            "- Linear-only baseline per token: attention q/k/v/o + MLP + LM head.",
            "- Only patched MLP projections are replaced by FFT block-circulant kernels.",
            "",
            "| B | Layers | Baseline model linear ops/token | Patched model linear ops/token | Total speedup vs baseline | Total op reduction |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in rows:
        lines.append(
            f"| {r.block_size} | {r.layers} | {r.baseline_model_linear_ops_per_token} | "
            f"{r.patched_model_linear_ops_per_token} | {r.total_model_speedup_vs_baseline:.3f}x | "
            f"{r.total_model_op_reduction_percent:.2f}% |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_table(rows: list[Row]) -> None:
    header = (
        f"{'B':>4} {'L':>4} {'Dense MACs':>14} {'Dense ops':>14} "
        f"{'FFT cmul':>12} {'FFT cadd':>12} {'rFFT':>8} {'iFFT':>8} {'FFT ops~':>12} {'Dense/FFT~':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.block_size:>4} {r.layers:>4} {r.dense_macs:>14} {r.dense_real_ops_mul_add:>14} "
            f"{r.fft_complex_multiplies:>12} {r.fft_accum_complex_adds:>12} {r.input_rfft_calls:>8} "
            f"{r.output_irfft_calls:>8} {r.approx_fft_real_ops:>12} {r.approx_fft_speedup_vs_dense_ops:>10.2f}x"
        )


def main() -> None:
    rows = build_rows()
    out_dir = Path(__file__).parent / "results_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "operation_counts.csv"
    md_path = out_dir / "operation_counts.md"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print_table(rows)
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
