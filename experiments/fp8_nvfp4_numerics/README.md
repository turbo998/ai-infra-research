# FP8 Fine-Grained Block Scaling vs Blackwell NVFP4 (`1x16 + RHT + SR`) Numerical Stability Model

## Overview

This deterministic Python standard-library experiment models the quantization error, non-outlier signal-to-quantization-noise ratio (`SQNR`), `WGRAD` inner-product fidelity, and Tensor Core accumulation error across five low-precision training recipes:

1. **Per-Tensor FP8 Delayed Scaling (`DelayedScaling`, E4M3)**: Single tensor-wide `amax` scale factor (`448 / delayed_amax`), vulnerable to subnormal underflow across normal channels when a single massive activation outlier (`10x`–`200x`) dominates `global_amax`.
2. **DeepSeek-V3 / DeepGEMM Fine-Grained `1x128` Block FP8 (`E4M3` + CUDA-Core FP32 Promotion)**: `1x128` activation/gradient tile scaling with periodic (`BLOCK_K=128`) promotion of Hopper WGMMA reduced-mantissa accumulators to full FP32 registers.
3. **OCP MXFP8 `1x32` (`E4M3` + `UE8M0` Power-of-Two Scale)**: Hardware micro-scaling over 32-element blocks with 8-bit exponent-only (`2^k`) scales.
4. **Blackwell NVFP4 (`NVFP4BlockScaling`, `E2M1` + `1x16 E4M3` + Global `FP32`)**: Two-level hierarchical block scaling (`s_global = global_amax / (448 * 6)`, `s_block = (block_amax / 6) / s_global` stored in `E4M3`), evaluated with and without **16x16 Random Hadamard Transform (`RHT`)**, **Stochastic Rounding (`SR`)**, and **4over6 adaptive block scaling**.

## Reproduction

```bash
python experiments/fp8_nvfp4_numerics/run.py
python experiments/fp8_nvfp4_numerics/run.py --check
```

## Key Findings (from `results.json`)

- **Per-Tensor FP8 Underflow vs `1x128` Block Isolation**: At `100x` outlier ratio, per-tensor FP8 Delayed Scaling loses `10.45 dB` of non-outlier SQNR (`21.49 dB`, dropping further to `15.47 dB` at `200x`), whereas DeepSeek `1x128` block scaling (`31.94 dB`) and MXFP8 `1x32` (`31.58 dB`) completely isolate the outlier to its local tile.
- **16x16 Random Hadamard Transform (`RHT`) on WGRAD**: Applying `H = (1/sqrt(16)) S_16 H_16` before columnwise NVFP4 quantization reduces the 16-element block crest factor from `3.048` to `1.410` and improves `WGRAD` inner-product SQNR at `100x` outlier ratio from `18.19 dB` (naive E2M1) to `35.06 dB` (`+16.87 dB` gain).
- **Stochastic Rounding (`SR`) Eliminates Systematic Gradient Bias**: For a sub-step gradient (`0.18` at block scale `0.5`), deterministic round-to-nearest (`RTN`) snaps every step to `0.25` (`38.89%` systematic bias), whereas hardware-accelerated NVFP4 Stochastic Rounding converges to `0.182251` (`1.25%` error over 4,096 steps).
- **WGMMA FP32 Promotion (`DeepGEMM` `BLOCK_K=128`)**: Promoting partial sums to FP32 every 128 MACs reduces 4,096-element dot-product accumulation relative error from `0.0346%` to `0.0056%` (`6.18x` error reduction).
