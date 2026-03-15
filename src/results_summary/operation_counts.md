# Operation Counts (Per Token)

Assumption: only patched MLP projections (gate, up, down) are counted.

| B | Layers | Dense MACs | Dense real ops (mul+add) | FFT complex mul | FFT complex add | rFFT calls | iFFT calls | Approx FFT real ops |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 135266304 | 270532608 | 1089792 | 1072764 | 300 | 408 | 9363960 |
| 128 | 1 | 135266304 | 270532608 | 536640 | 519870 | 150 | 204 | 5052540 |
| 256 | 1 | 135266304 | 270532608 | 266256 | 249615 | 75 | 102 | 3003006 |
| 64 | 8 | 1082130432 | 2164260864 | 8718336 | 8582112 | 2400 | 3264 | 74911680 |
| 128 | 8 | 1082130432 | 2164260864 | 4293120 | 4158960 | 1200 | 1632 | 40420320 |
| 256 | 8 | 1082130432 | 2164260864 | 2130048 | 1996920 | 600 | 816 | 24024048 |

## Relative factors vs Dense

- `Dense/FFT approx real-ops factor` > 1 means FFT path uses fewer estimated real operations.
- `Dense MAC / FFT complex-mul factor` is a lower-level kernel ratio (ignores FFT call overhead).

| B | Layers | Dense/FFT approx real-ops factor | Dense MAC / FFT complex-mul factor |
|---:|---:|---:|---:|
| 64 | 1 | 28.89x | 124.12x |
| 128 | 1 | 53.54x | 252.06x |
| 256 | 1 | 90.09x | 508.03x |
| 64 | 8 | 28.89x | 124.12x |
| 128 | 8 | 53.54x | 252.06x |
| 256 | 8 | 90.09x | 508.03x |

## System-level estimate vs full model baseline

Assumptions for this section:
- Llama-style decoder with 32 transformer layers.
- Linear-only baseline per token: attention q/k/v/o + MLP + LM head.
- Only patched MLP projections are replaced by FFT block-circulant kernels.

| B | Layers | Baseline model linear ops/token | Patched model linear ops/token | Total speedup vs baseline | Total op reduction |
|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 13214154752 | 12952986104 | 1.020x | 1.98% |
| 128 | 1 | 13214154752 | 12948674684 | 1.021x | 2.01% |
| 256 | 1 | 13214154752 | 12946625150 | 1.021x | 2.02% |
| 64 | 8 | 13214154752 | 11124805568 | 1.188x | 15.81% |
| 128 | 8 | 13214154752 | 11090314208 | 1.192x | 16.07% |
| 256 | 8 | 13214154752 | 11073917936 | 1.193x | 16.20% |