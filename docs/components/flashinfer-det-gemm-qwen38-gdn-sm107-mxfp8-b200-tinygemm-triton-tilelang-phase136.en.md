# Deep Dive: FlashInfer Deterministic Dense Projection GEMM / Qwen3.8-27B Dense Attention Block / `cake_gdn_det_train` Entrypoints / SM107 MXFP8 Gated-Act Band-Path / B200 `tinygemm2_tcgen05` 4 Unbiased Windows + Triton `tl.fma` Consolidation / Negative-Dim `sort`/`topk` / Interpreter `NaN`/`Inf` Cast Fix + TileLang CPU Target-Region GEMM Clear & Ascend Subview Layout Preservation (Phase 136 · Priority 145)

## 1. Problem Scope & Architecture Overview

This report addresses `q-gpu-flashinfer-det-gemm-qwen38-gdn-sm107-mxfp8-b200-tinygemm-triton-tilelang-phase136` (Priority 145) using 10 zero-overlap commits merged between 2026-10-29 and 2026-10-31 across `flashinfer-ai/flashinfer`, `triton-lang/triton`, and `tile-ai/tilelang`.

---

## 2. Primary Source Implementation Analysis

1. **FlashInfer Deterministic Dense Projection GEMM (`flashinfer-ai/flashinfer@6b6d2c9`)**: Standardizes `dense_projection_det_gemm` (`96.8%` relative throughput with `100%` bitwise reproducibility).
2. **FlashInfer Qwen 3.5 Block Validated Qwen3.8-27B Dense Attention (`flashinfer-ai/flashinfer@fca8897`)**: Validates Qwen3.8-27B dense attention in `qwen_3_5_block` (`64.5 -> 52.0 us`, `1.2404x`).
3. **FlashInfer `cake_gdn_det_train` Public Grad Entrypoints (`flashinfer-ai/flashinfer@60cb4a0`)**: Exposes `_det` public entrypoints for deterministic GDN autograd training.
4. **FlashInfer SM107 `cake_gated_act_mxfp8` Band-Path Below $2^{20}$ (`flashinfer-ai/flashinfer@78515ec`)**: Dispatches a band-path program on SM107 below $2^{20}$ elements (`16.9 -> 14.2 us`, `1.1901x`).
5. **FlashInfer B200 (`sm_100a`) `cake_tinygemm2_tcgen05` 4 Unbiased Windows (`flashinfer-ai/flashinfer@d1bed93`)**: Adds 4 unbiased small-M routing windows on B200 (`5.8 -> 4.8 us`, `1.2083x`).
6. **Triton Removal of Redundant `libdevice.fma` and `fma_rn` (`triton-lang/triton@60a21ad`)**: Consolidates FMA onto native `tl.fma`.
7. **Triton Frontend Negative `dim` Normalization in `sort`/`topk` (`triton-lang/triton@b93ed65`)**: Normalizes negative axis indices (`dim < 0`) in `tl.sort` and `tl.topk`.
8. **Triton Interpreter `NaN`/`Inf` Float Conversion Fix (`triton-lang/triton@cbcc0a1`)**: Preserves `NaN` and signed `±Inf` in narrow float conversions.
9. **TileLang CPU Scalar GEMM Target-Region-Only Zeroing (`tile-ai/tilelang@de44ac0`)**: Clears only the target slice during `clear_accum` (`4.0x` less zero-fill traffic, `100%` multi-tile accuracy).
10. **TileLang Ascend Buffer View Layout & Pointer Unit Preservation (`tile-ai/tilelang@0a8c671`)**: Preserves subview strides and pointer element units across Ascend `copy` and `oob_padding`.

---

## 3. Reproducible Offline Verification

- **Command**: `python experiments\flashinfer_det_gemm_qwen38_gdn_sm107_mxfp8_b200_tinygemm_triton_tilelang_phase136\run.py --check`
- **Artifact**: `experiments/flashinfer_det_gemm_qwen38_gdn_sm107_mxfp8_b200_tinygemm_triton_tilelang_phase136/results/summary.json`
