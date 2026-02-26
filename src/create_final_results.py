"""
Script to consolidate all benchmark results into a clean summary table for the seminar.
Generates a final results table in Markdown and CSV format.
"""

import csv
from pathlib import Path

# Define the data manually from the CSV files we analyzed
# This ensures consistency and allows us to annotate the conditions

results = [
    # Configuration: Original Model (Baseline)
    {
        "config": "Original (Baseline)",
        "layers_patched": 0,
        "block_size": "-",
        "calibrated": "-",
        "cos_sim": 1.0,
        "kl_div": 0.0,
        "top1_acc": 1.0,
        "top5_overlap": 1.0,
        "prefill_ms": 499,
        "tok_per_s": 101,
        "source": "bench_baseline_orig.csv"
    },
    
    # Configuration: 1 Layer, No Calibration
    {
        "config": "1 Layer, No Calib",
        "layers_patched": 1,
        "block_size": 64,
        "calibrated": "No",
        "cos_sim": 0.890,
        "kl_div": 1.237,
        "top1_acc": 0.55,
        "top5_overlap": 0.55,
        "prefill_ms": 480,
        "tok_per_s": 40,
        "source": "bench_final_calib_nocal.csv"
    },
    {
        "config": "1 Layer, No Calib",
        "layers_patched": 1,
        "block_size": 128,
        "calibrated": "No",
        "cos_sim": 0.890,
        "kl_div": 1.236,
        "top1_acc": 0.55,
        "top5_overlap": 0.55,
        "prefill_ms": 473,
        "tok_per_s": 41,
        "source": "bench_final_calib_nocal.csv"
    },
    {
        "config": "1 Layer, No Calib",
        "layers_patched": 1,
        "block_size": 256,
        "calibrated": "No",
        "cos_sim": 0.890,
        "kl_div": 1.236,
        "top1_acc": 0.55,
        "top5_overlap": 0.55,
        "prefill_ms": 475,
        "tok_per_s": 41,
        "source": "bench_final_calib_nocal.csv"
    },
    
    # Configuration: 1 Layer, With Calibration
    {
        "config": "1 Layer, Calibrated",
        "layers_patched": 1,
        "block_size": 64,
        "calibrated": "Yes",
        "cos_sim": 0.815,
        "kl_div": 0.326,
        "top1_acc": 1.0,
        "top5_overlap": 0.51,
        "prefill_ms": 539,
        "tok_per_s": 94,
        "source": "bench_eval_long_calib_L1.csv"
    },
    {
        "config": "1 Layer, Calibrated",
        "layers_patched": 1,
        "block_size": 128,
        "calibrated": "Yes",
        "cos_sim": 0.857,
        "kl_div": 0.662,
        "top1_acc": 1.0,
        "top5_overlap": 0.53,
        "prefill_ms": 495,
        "tok_per_s": 102,
        "source": "bench_eval_long_calib_L1.csv"
    },
    {
        "config": "1 Layer, Calibrated",
        "layers_patched": 1,
        "block_size": 256,
        "calibrated": "Yes",
        "cos_sim": 0.824,
        "kl_div": 3.103,
        "top1_acc": 0.15,
        "top5_overlap": 0.19,
        "prefill_ms": 498,
        "tok_per_s": 101,
        "source": "bench_eval_long_calib_L1.csv"
    },
    
    # Configuration: 4 Layers, No Calibration (problematic)
    {
        "config": "4 Layers, No Calib",
        "layers_patched": 4,
        "block_size": 64,
        "calibrated": "No",
        "cos_sim": 0.318,
        "kl_div": 5.771,
        "top1_acc": 0.0,
        "top5_overlap": 0.0,
        "prefill_ms": 575,
        "tok_per_s": 34,
        "source": "bench_final_nocal_4layer.csv"
    },
    {
        "config": "4 Layers, No Calib",
        "layers_patched": 4,
        "block_size": 128,
        "calibrated": "No",
        "cos_sim": 0.318,
        "kl_div": 5.772,
        "top1_acc": 0.0,
        "top5_overlap": 0.0,
        "prefill_ms": 500,
        "tok_per_s": 39,
        "source": "bench_final_nocal_4layer.csv"
    },
    
    # Configuration: 8 Layers, With Calibration (problematic on eval data)
    {
        "config": "8 Layers, Calibrated",
        "layers_patched": 8,
        "block_size": 64,
        "calibrated": "Yes",
        "cos_sim": 0.793,
        "kl_div": 3.299,
        "top1_acc": 0.0,
        "top5_overlap": 0.15,
        "prefill_ms": 626,
        "tok_per_s": 80,
        "source": "bench_eval_long_calib_L8.csv"
    },
    {
        "config": "8 Layers, Calibrated",
        "layers_patched": 8,
        "block_size": 128,
        "calibrated": "Yes",
        "cos_sim": 0.804,
        "kl_div": 3.125,
        "top1_acc": 0.0,
        "top5_overlap": 0.20,
        "prefill_ms": 587,
        "tok_per_s": 86,
        "source": "bench_eval_long_calib_L8.csv"
    },
]

def create_summary():
    # Column names for the summary
    cols = ["config", "layers_patched", "block_size", "calibrated", 
            "cos_sim", "kl_div", "top1_acc", "top5_overlap", "tok_per_s"]
    
    nice_names = [
        "Configuration", "Layers", "Block Size", "Calibrated",
        "Cosine Sim", "KL Div", "Top-1 Acc", "Top-5 Overlap", "Tokens/s"
    ]
    
    # Extract only the columns we need
    rows = []
    for r in results:
        rows.append([r[c] for c in cols])
    
    return nice_names, rows

def create_markdown_table(headers, rows):
    """Create a Markdown table string."""
    lines = []
    
    # Header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    # Rows
    for row in rows:
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    
    return "\n".join(lines)

def create_csv_table(headers, rows):
    """Create CSV string."""
    lines = [",".join(headers)]
    for row in rows:
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        lines.append(",".join(cells))
    return "\n".join(lines)

def print_table(headers, rows):
    """Print a nice ASCII table."""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if isinstance(val, float):
                s = f"{val:.3f}"
            else:
                s = str(val)
            widths[i] = max(widths[i], len(s))
    
    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        cells = []
        for i, val in enumerate(row):
            if isinstance(val, float):
                s = f"{val:.3f}"
            else:
                s = str(val)
            cells.append(s.ljust(widths[i]))
        print(" | ".join(cells))

def main():
    output_dir = Path(__file__).parent / "results_summary"
    output_dir.mkdir(exist_ok=True)
    
    headers, rows = create_summary()
    
    # Save CSV
    csv_path = output_dir / "final_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(create_csv_table(headers, rows))
    print(f"Saved: {csv_path}")
    
    # Create Markdown
    md_table = create_markdown_table(headers, rows)
    
    # Create full report
    report = f"""# Llama-2 FFT Block-Circulant Results Summary

## Overview
This report summarizes the results of replacing dense MLP layers in Llama-2-7B 
with FFT-based block-circulant approximations.

## Key Findings

### 1. Single Layer Patching Works Well
- **Best configuration**: 1 layer patched with calibration, block size 64 or 128
- Achieves **100% Top-1 accuracy** with KL divergence < 0.7
- Cosine similarity to original: ~0.85

### 2. More Layers = More Errors
- 4+ layers without calibration: catastrophic (cos < 0.32)
- 8 layers with calibration: still poor on held-out data (Top-1 = 0%)
- Error accumulation makes multi-layer patching challenging

### 3. Calibration Helps (for 1 layer)
- Without calibration: Top-1 ~55%, KL ~1.24
- With calibration (B=64): Top-1 ~100%, KL ~0.33
- Block size 256 shows degradation even with calibration

## Results Table

{md_table}

## Metrics Explanation
- **Cosine Sim**: Cosine similarity of last-token logits to original model (1.0 = identical)
- **KL Div**: KL divergence from original model's distribution (0 = identical)  
- **Top-1 Acc**: Fraction of prompts where predicted next token matches original
- **Top-5 Overlap**: Overlap of top-5 predicted tokens with original

## Recommendations for Practical Use
1. Patch only 1-2 layers for acceptable quality
2. Use block size 64 or 128 (256 is too coarse)
3. Calibration is essential for maintaining prediction quality
4. The theoretical O(n log n) speedup is offset by FFT overhead in practice

## Notes
- Model: Llama-2-7B-hf
- Evaluation: 20 quality prompts, max length 256 tokens
- Hardware: CUDA GPU with ~13GB VRAM
"""
    
    md_path = output_dir / "final_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved: {md_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60 + "\n")
    print_table(headers, rows)
    print("\n" + "="*60)
    
    # Key takeaways
    print("\nKEY TAKEAWAYS:")
    print("✓ 1 Layer + Calibration + B=64: Top-1=100%, KL=0.33")
    print("✓ 1 Layer + Calibration + B=128: Top-1=100%, KL=0.66")
    print("✗ 4+ Layers: Error accumulation makes this impractical")
    print("✗ B=256: Too coarse, quality degrades even with 1 layer")

if __name__ == "__main__":
    main()
