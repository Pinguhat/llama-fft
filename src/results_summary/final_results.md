# Llama-2 FFT Block-Circulant Results Summary

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

| Configuration | Layers | Block Size | Calibrated | Cosine Sim | KL Div | Top-1 Acc | Top-5 Overlap | Tokens/s |
|---|---|---|---|---|---|---|---|---|
| Original (Baseline) | 0 | - | - | 1.000 | 0.000 | 1.000 | 1.000 | 101 |
| 1 Layer, No Calib | 1 | 64 | No | 0.890 | 1.237 | 0.550 | 0.550 | 40 |
| 1 Layer, No Calib | 1 | 128 | No | 0.890 | 1.236 | 0.550 | 0.550 | 41 |
| 1 Layer, No Calib | 1 | 256 | No | 0.890 | 1.236 | 0.550 | 0.550 | 41 |
| 1 Layer, Calibrated | 1 | 64 | Yes | 0.815 | 0.326 | 1.000 | 0.510 | 94 |
| 1 Layer, Calibrated | 1 | 128 | Yes | 0.857 | 0.662 | 1.000 | 0.530 | 102 |
| 1 Layer, Calibrated | 1 | 256 | Yes | 0.824 | 3.103 | 0.150 | 0.190 | 101 |
| 4 Layers, No Calib | 4 | 64 | No | 0.318 | 5.771 | 0.000 | 0.000 | 34 |
| 4 Layers, No Calib | 4 | 128 | No | 0.318 | 5.772 | 0.000 | 0.000 | 39 |
| 8 Layers, Calibrated | 8 | 64 | Yes | 0.793 | 3.299 | 0.000 | 0.150 | 80 |
| 8 Layers, Calibrated | 8 | 128 | Yes | 0.804 | 3.125 | 0.000 | 0.200 | 86 |

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
