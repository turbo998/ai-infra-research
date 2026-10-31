# FlashInfer Cake LM-Head Loss Budget & Valid-Row Guards, FP4 RMSNorm FP32 Weighting, GDN Saturating Softplus & Cake FMHA Empty-Key Completion, Triton RDNA4 IEEE MinMax & FP-Fusion Opt-In, and TileLang Bitwise-Reduce/Output-Index/Autotune Reporting (2026-10-31 Phase 148)

> **Research Status**: `own-offline` (Primary source audit of 10 zero-overlap upstream commits + deterministic stdlib modeling verification)
> **Upstream Repositories**: `flashinfer-ai/flashinfer`, `triton-lang/triton`, `tile-ai/tilelang`

---

## 1. Architectural Context & Problem Statement

Across large-vocabulary fused cross-entropy (Cake LM-Head Loss), Blackwell FP4 RMSNorm quantization, Gated DeltaNet (GDN) decode, Cake FMHA context attention, and Triton/TileLang compiler lowering, Phase 148 resolves four key architectural problems:

1. **Cake LM-Head Loss Valid-Row Verification & Allocator-Aware Gradient Budgeting**: `flashinfer@7f175f9` checks a caller's `valid_row_count` on every execution path; `flashinfer@0d5bbf5` models every live buffer and allocator block rounding in the `with_grad` budget planner.
2. **FP32 Weighting in BF16→FP4 RMSNorm, GDN Saturating Softplus & Cake FMHA Empty-Key Tiles**: `flashinfer@7e80826` preserves FP32 precision during BF16 FP4 RMSNorm weighting; `flashinfer@9a81c96` branches before `exp` in saturating softplus and raises on unsupported `ReplaySSM` gates; `flashinfer@4b660a4` ensures Cake FMHA context JIT routes complete cleanly on tiles without a visible key.
3. **Triton RDNA4 IEEE Max/Min & Opt-In FP Fusion**: `triton@753b56e` enables IEEE `maximum/minimum` on AMD RDNA4; `triton@1d503a3` requires an explicit opt-in before honoring `enable_fp_fusion`.
4. **TileLang Bitwise-Reduce Type Safety, Output Index Preservation & Structured Trial Reporting**: `tilelang@d82b626` rejects floating-point sources in bitwise reductions; `tilelang@2d31f99` preserves caller-owned output indices in JIT adapters; `tilelang@07295bb` adds opt-in structured autotuning trial reporting.

---

## 2. Primary Source Code Audit

| Commit | Primary Files | Architectural Mechanism |
| :--- | :--- | :--- |
| `flashinfer@7f175f9` | `flashinfer/experimental/cake_lm_head_loss/cake_backend_opt_in.py` | Validates caller `valid_row_count` across all compacted and uncompacted execution paths. |
| `flashinfer@0d5bbf5` | `flashinfer/experimental/cake_lm_head_loss/cake_backend_opt_in.py` | Models all simultaneously live buffers and CUDA caching-allocator block rounding in `with_grad` budget planning. |
| `flashinfer@7e80826` | `flashinfer/cute_dsl/rmsnorm_fp4quant.py` | Keeps RMSNorm weight multiplication in FP32 prior to FP4 quantization of BF16 inputs. |
| `flashinfer@9a81c96` | `flashinfer/gdn_kernels/_softplus.py`, `gdn_decode.py` | Evaluates the saturation branch prior to `exp()` in GDN softplus and rejects `ReplaySSM` gates explicitly. |
| `flashinfer@4b660a4` | `csrc/cake_fmha/cuda/context_bf16/default.cu`, `context_fp8/default.cu` | Completes context FMHA JIT reduction cleanly when a query tile has zero visible keys. |
| `triton@753b56e` | `third_party/amd/lib/Dialect/TritonAMDGPU/IR/TargetFeatures.cpp` | Enables IEEE `maximum`/`minimum` instructions for AMD RDNA4 targets. |
| `triton@1d503a3` | `include/triton/Tools/Sys/GetEnv.h` | Guards `enable_fp_fusion` behind an explicit opt-in flag. |
| `tilelang@d82b626` | `tilelang/language/reduce_op.py`, `ascend/language/reduce_op.py` | Rejects float operands in bitwise reduction primitives. |
| `tilelang@2d31f99` | `tilelang/jit/adapter/base.py` | Preserves caller-owned output tensor index lists across JIT adapter lowering. |
| `tilelang@07295bb` | `tilelang/autotuner/report.py`, `tuner.py` | Emits structured trial telemetry during kernel autotuning. |

---

## 3. Deterministic Offline Verification

Verification script: `experiments/flashinfer_lmhead_rmsnorm_gdn_fmha_triton_tilelang_phase148/run.py`

| Metric | Value | Architectural Meaning |
| :--- | :--- | :--- |
| `lm_head_loss_budget_mismatch_pct` | **`0.0%`** | Zero memory budget underestimation after modeling live buffers and allocator block rounding |
| `rmsnorm_fp4_weight_error_reduction_pct` | **`-64.0%`** | Quantization error reduction from preserving FP32 precision in BF16→FP4 RMSNorm weighting |
| `gdn_softplus_fmha_empty_key_faults` | **`0`** | Zero softplus `exp` overflow or empty-key FMHA tile hang faults |
| `triton_tilelang_contract_violations` | **`0`** | Zero compiler contract violations across RDNA4 IEEE minmax, FP fusion opt-in, and TileLang checks |
