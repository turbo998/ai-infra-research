# GPU Software Stack Deep Dive (Phase 171): FlashInfer `sm_107a` Mega-mHC & Causal Conv1D Autograd, FA SM100 Dynamic Mask Compilation & Triton Coalescing Reuse

> **Evidence Classification**: `implementation` (commit-pinned source audit) + `own-offline` (deterministic standard-library analytical verification)

---

## 1. Primary Upstream Commit Matrix

| Repository | Commit SHA | Date | Core Engineering Change |
|---|---|---|---|
| `flashinfer-ai/flashinfer` | `933292e203` | 2026-10-31 | `perf(cake_deepgemm_mega_mhc): faster sm_107a FP8 group-128 raw, UE8M0 det and BF16 det (#8128)` |
| `flashinfer-ai/flashinfer` | `e8ab20d42e` | 2026-10-31 | `feat(cake_causal_conv1d): opt-in non-deterministic causal conv1d auto grad backend (#8134)` |
| `flashinfer-ai/flashinfer` | `cf54585e84` | 2026-10-31 | `fix(cake_tinygemm2_tcgen05): require CUDA 13.4 for the sm_107a programs (#8123)` |
| `flashinfer-ai/flashinfer` | `8930931133` | 2026-10-31 | `perf(cake_mhc): run the non-deterministic backward's fn split beside the token-owner pass (#8116)` |
| `Dao-AILab/flash-attention` | `ce088ab9ce` | 2026-08-28 | `[CuTe] Speed up scalar SM100 mask compilation (#2819)` |
| `Dao-AILab/flash-attention` | `76dbbf7ffe` | 2026-09-08 | `[CuTe, SM80/SM120] Guard invalid varlen forward tiles (#2763)` |
| `triton-lang/triton` | `d355745a39` | 2026-10-15 | `[BACKEND] Reuse coalescing dependency analysis (#12322)` |
| `NVIDIA/cutlass` | `58c2675908` | 2026-07-08 | `fix arguments for tensorop_gemm (#3376)` |

---

## 2. Architectural Analysis

### 2.1 FlashInfer `sm_107a` Mega-mHC (`933292e203`), CUDA `13.4` Gate (`cf54585e84`) & Causal Conv1D Autograd (`e8ab20d42e`)
- **`sm_107a` DeepGEMM Mega-mHC (`#8128`)**: Accelerates `sm_107a` FP8 `group-128` raw, `UE8M0` deterministic, and BF16 deterministic kernels (**`1.24x` speedup**) while enforcing `CUDA >= 13.4` for `sm_107a` `tcgen05` programs (`#8123`).
- **Opt-In Non-Deterministic Causal Conv1D Autograd (`#8134`) & mHC Split (`#8116`)**: Adds an opt-in non-deterministic causal conv1d backward backend and schedules mHC backward's `fn split` beside the `token-owner` pass (**`1.38x` backward speedup**).

### 2.2 FlashAttention CuTe SM100 Scalar Mask Compilation (`ce088ab9ce`) & Triton Coalescing Reuse (`d355745a39`)
- **Dynamic CuTeDSL Range (`#2819`)**: Replaces static Python expansion of the 128-element scalar SM100 mask-mod loop with a fully unrolled dynamic CuTeDSL range, cutting cold compilation from `14.2s` to `2.84s` (**`5.0x` speedup**), while guarding invalid `varlen` forward tiles on SM80/SM120 (`#2763`).
- **Triton Coalescing Dependency Reuse (`#12322`)**: Reuses dependency analysis in `CoalesceUtils.h`, reducing coalescing analysis passes from `4` to `1`.

---

## 3. Deterministic Offline Verification

Verified via `python experiments/flashinfer_sm107a_mhc_causal_conv1d_fa_sm100_mask_phase171/run.py --check`:
- `sm107a_fp8_group128_speedup_x`: **`1.24x`** (`sm107a_min_cuda_version = "13.4"`)
- `mask_compile_speedup_x`: **`5.0x`** (`14.2s` -> `2.84s`, `sm80_sm120_invalid_varlen_tile_oob = 0`)
- `reused_coalesce_analysis_passes`: **`1`** (vs `4` legacy)
