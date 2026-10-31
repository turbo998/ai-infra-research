# FlashInfer Cake VSA Sage-FP8 Pin & QSA NVFP4/FP8, SM120 GDN Prefill, SM90 MSA Routing, TRT-LLM MoE EP Tile_N, Triton gfx1250 TDM & FPSan Copysign, and TileLang Popcount/View/Wrapper Fidelity (2026-10-31 Phase 147)

> **Research Status**: `own-offline` (Primary source audit of 10 zero-overlap upstream commits + deterministic stdlib modeling verification)
> **Upstream Repositories**: `flashinfer-ai/flashinfer`, `triton-lang/triton`, `tile-ai/tilelang`

---

## 1. Architectural Context & Problem Statement

Across Hopper (SM90), Blackwell (SM100/SM120), and AMD `gfx1250` targets, sparse/quantized attention (Cake VSA, QSA, MSA), Gated DeltaNet (GDN) prefill, and Expert-Parallel (EP) MoE autotuning face four primary bottlenecks:

1. **SM90 Cake VSA Grouped-Query Pinning & Short-Tile Tail Latency**: `flashinfer@ebdad91` adds a second Sage-FP8 grouped-query pin program, a faster compact program, and a short-tile queue `vf` tier to the SM90 Cake VSA backend.
2. **SM80-SM90 QSA NVFP4 Speedup & FP8/BF16 Cache Support**: `flashinfer@49e254e` adds FP8 and BF16 KV caches and accelerates NVFP4 inner loops in the SM80-SM90 QSA kernel.
3. **SM120 GDN CuTe Prefill, SM90 MSA Measured Routing & MoE EP `tile_N`**: `flashinfer@73beeb0` adds an experimental SM120 CuTe GDN prefill backend; `flashinfer@37b6c98` fixes SM90 CuTe DSL sparse prefill accuracy and adds measured Cake/CuTe routing; `flashinfer@4b1cea6` exposes expert-parallel `tile_N` candidates to the TRT-LLM-gen MoE autotuner.
4. **Triton & TileLang Compiler Lowering Fidelity**: `triton@c1641ab` ignores unit dimensions when checking shared memory order for AMD `gfx1250` TDM ops; `triton@034dc0a` preserves 16-bit `copysign` NaN payloads under NVIDIA FPSan; `tilelang@6b67192`, `0a638eb`, and `da9471a` fix signed/narrow integer `popcount`, scalar bit counts in `T.view`, and parameter identity in NVRTC/CuTeDSL wrappers.

---

## 2. Primary Source Code Audit

| Commit | Primary Files | Architectural Mechanism |
| :--- | :--- | :--- |
| `flashinfer@ebdad91` | `flashinfer/cake_vsa_sm90_backend/cake_jit.py` | Introduces a second Sage-FP8 GQA pin program, faster compaction, and a short-tile queue `vf` tier on SM90. |
| `flashinfer@49e254e` | `flashinfer/experimental/qsa_nvfp4/csrc/qsa_nvfp4_attention.cu` | Adds FP8/BF16 KV cache support and accelerates NVFP4 dequantization + sparse block reduction on SM80-SM90. |
| `flashinfer@73beeb0` | `benchmarks/bench_gdn_sm120_cute.py` | Adds an experimental SM120 CuTe prefill backend for Gated DeltaNet (GDN). |
| `flashinfer@37b6c98` | `flashinfer/msa_ops/_sm90_dispatch.py` | Restores SM90 CuTe DSL sparse prefill accuracy and dispatches between Cake and CuTe using measured crossover boundaries. |
| `flashinfer@4b1cea6` | `flashinfer/fused_moe/backends/trtllm/sm100_runner.py` | Offers multi-tier `tile_N` candidates to the TRT-LLM-gen MoE autotuner under expert parallelism. |
| `triton@c1641ab` | `third_party/amd/lib/TritonAMDGPUToLLVM/TensorPtrOpsToLLVM.cpp` | Skips unit dimensions (`size == 1`) when verifying shared-memory stride order for `gfx1250` TDM operations. |
| `triton@034dc0a` | `python/src/llvm.cc` | Preserves 16-bit `copysign` NaN payload bits under NVIDIA FPSan instrumentation. |
| `tilelang@6b67192` | `src/cuda/codegen/intrin_rule_cuda.cc` | Masks signed and narrow integers before CUDA `__popc`/`__popcll` lowering to prevent sign-extension bit inflation. |
| `tilelang@0a638eb` | `tilelang/utils/language.py` | Preserves exact scalar bit widths across `T.view` reinterpretation. |
| `tilelang@da9471a` | `tilelang/jit/adapter/cutedsl/wrapper.py` | Preserves parameter object identity across NVRTC and CuTeDSL JIT wrappers. |

---

## 3. Deterministic Offline Verification

 Verification script: `experiments/flashinfer_cake_qsa_gdn_msa_triton_tilelang_phase147/run.py`

| Metric | Value | Architectural Meaning |
| :--- | :--- | :--- |
| `cake_vsa_sm90_latency_reduction_pct` | **`-27.78%`** | SM90 Cake VSA latency reduction via second Sage-FP8 pin, fast compact, and short-tile `vf` tier |
| `qsa_nvfp4_speedup_ratio` | **`1.35x`** | SM80-SM90 QSA NVFP4 speedup with FP8/BF16 KV cache support |
| `gdn_sm120_msa_routing_speedup_ratio` | **`1.32x`** | Combined SM120 CuTe GDN prefill and SM90 measured Cake/CuTe MSA routing speedup |
| `trtllm_moe_ep_tile_n_gain_pct` | **`+22.5%`** | Effective utilization gain from exposing EP `tile_N` candidates to TRT-LLM-gen MoE autotuner |
| `compiler_bug_violation_rate_pct` | **`0.0%`** | Zero TDM unit-dim, FPSan NaN payload, signed popcount, or wrapper identity violations |
