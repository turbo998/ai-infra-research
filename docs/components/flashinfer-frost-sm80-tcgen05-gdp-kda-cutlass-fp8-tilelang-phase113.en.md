# FlashInfer FROST SM80 Grouped MoE, Blackwell `cake_tinygemm2_tcgen05`, GDP Decode Preload, DeepGEMM `n_l2_group=8` Cluster Alignment, KDA Segmented Carry, Blackwell Cluster Softmax & CUTLASS / TileLang Analysis (Phase 113 · Priority 122)

> **中文版**: [flashinfer-frost-sm80-tcgen05-gdp-kda-cutlass-fp8-tilelang-phase113.md](flashinfer-frost-sm80-tcgen05-gdp-kda-cutlass-fp8-tilelang-phase113.md)  
> **Question ID**: `flashinfer-frost-sm80-tcgen05-gdp-kda-cutlass-fp8-tilelang-phase113` · **Priority**: `122` · **Track**: Track A (GPU/HBM/Kernel Stack)  
> **Deterministic Experiment**: [`experiments/flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang_phase113/run.py`](../../experiments/flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang_phase113/run.py) (`--check` passed)

---

## 1. First-Hand Source Matrix

| # | Source ID | Repository & Commit | Date | Evidence Tier | Key Code Path & Architectural Change |
|---|---|---|---|---|---|
| 1 | `flashinfer-moe-frost-decode-sm80-typed-activations-2026` | [`flashinfer-ai/flashinfer@0e406f333b`](https://github.com/flashinfer-ai/flashinfer/commit/0e406f333bde1eb186023a1d28e882a235641a10) | `2026-10-25` | `implementation` + `author-benchmark` | Extends `FrostLowLatencyConfig` to Ampere SM80 (`128-bit cp.async` + warp MMA, `--gpu-arch sm_80` without `'a'`) and widens SM90/SM120 grouped paths to FP16 and `SwiGLU`/`GeGLU`/`ReLU2`/`SiLU` |
| 2 | `flashinfer-cake-tinygemm2-tcgen05-bf16-sm100-103-107-2026` | [`flashinfer-ai/flashinfer@8d9c839db0`](https://github.com/flashinfer-ai/flashinfer/commit/8d9c839db08d157d1bb4bd19645266813ef57d6a) | `2026-10-25` | `implementation` + `author-benchmark` | Adds `cake_tinygemm2_tcgen05` with 12 `tcgen05` programs per arch across `sm_100a`/`sm_103a`/`sm_107a` (`36` total), cutting B200 `4×2880×2880` bias-free latency from `8.29 -> 7.26 us` (**`1.1419x` speedup**) |
| 3 | `flashinfer-cake-gdp-preload-dense-row-t1-decode-sm100-2026` | [`flashinfer-ai/flashinfer@9f6ba5a820`](https://github.com/flashinfer-ai/flashinfer/commit/9f6ba5a820714fa30dc88bf6f6a3775819a21f40) | `2026-10-25` | `implementation` + `author-benchmark` | Preloads dense `v`/`beta` rows at the `k` load site in the one-row-per-thread $T=1$ Cake GDP decode kernel on SM100/SM103 (**up to `1.07x` on GB300, `1.06x` on B200**) |
| 4 | `flashinfer-cake-batch-deepgemm-fp8-even-nl2-group-k7168-2026` | [`flashinfer-ai/flashinfer@4d018bcd12`](https://github.com/flashinfer-ai/flashinfer/commit/4d018bcd12d4fc8b24f6838d0dc6c4c63a195322) | `2026-10-25` | `implementation` + `author-benchmark` | Switches `cake_batch_deepgemm_fp8` at `K=7168, expected_m<=1` from `n_l2_group=1` to even `n_l2_group=8` so 2-CTA clusters share one 32-row M block (**reducing bad elements on GB300 from `97,355` to `0`**) |
| 5 | `flashinfer-cake-kda-fused-gate-segmented-16head-carry-2026` | [`flashinfer-ai/flashinfer@762b237e15`](https://github.com/flashinfer-ai/flashinfer/commit/762b237e15bdca33368889186bc26a234625e50f) | `2026-10-25` | `implementation` + `author-benchmark` | Fuses the KDA backward `gate_epilogue` into `glm_intra_v14` on SM103 (`4 -> 3` tail launches), enables segmented 16-head recurrence on SM103/SM107, and uses two-level segment carry on SM100/SM107 |
| 6 | `flashinfer-cake-softmax-blackwell-cached-cluster-persistent-2026` | [`flashinfer-ai/flashinfer@510e514b25`](https://github.com/flashinfer-ai/flashinfer/commit/510e514b25f1379cc81c45cc90b82ab7d68d9b00) | `2026-10-25` | `implementation` + `author-benchmark` | Regenerates 7 Blackwell row-caching cluster softmax programs with push max/sum exchange and adds 2 persistent programs above `2.5x` resident clusters (**`1.283x` B200 / `1.263x` GB300 geomean across 41 shapes**) |
| 7 | `cutlass-epilogue-per-group-norm-constant-index-fix-2026` | [`NVIDIA/cutlass@ad9bd53bda`](https://github.com/NVIDIA/cutlass/commit/ad9bd53bda6ffddbfcf6c6b7ddfd950853ee339c) | `2026-07-18` | `implementation` | Captures the group index before zeroing batch coordinate `l` in SM100/SM120 TMA warp-specialized block-scale-factor epilogues, eliminating group-0 norm constant aliasing |
| 8 | `cutlass-hopper-cutedsl-fp8-gemm-gelu-epilogue-2026` | [`NVIDIA/cutlass@a931725f1d`](https://github.com/NVIDIA/cutlass/commit/a931725f1ddc2f1dc171c22e362c4338179c6d42) | `2026-07-14` | `implementation` | Adds parameterized Hopper CuTeDSL persistent FP8 GEMM with fused GELU epilogues |
| 9 | `tilelang-ascend-pto-emit-ramp-typed-vectors-2026` | [`tile-ai/tilelang@d7a28bd78b`](https://github.com/tile-ai/tilelang/commit/d7a28bd78b544dcdfba2f8791808236f977f0c94) | `2026-10-19` | `implementation` | Emits Ascend PTO `Ramp` expressions via `vector_from_list` as a single typed vector argument instead of unparenthesized tuple lanes |
| 10 | `tilelang-ascend-reset-rng-state-between-kernels-2026` | [`tile-ai/tilelang@44405ee548`](https://github.com/tile-ai/tilelang/commit/44405ee54834167794bf7cb3fa334e01c2d06745) | `2026-10-16` | `implementation` | Resets Ascend codegen RNG state between device kernels |

---

## 2. Deterministic Verification Summary (`own-offline`)

| Metric | Baseline / Before | Phase 113 Verified | Delta / Takeaway |
|---|---|---|---|
| FROST arch compiler flag (`SM80` vs `SM90`/`SM120`) | Invalid `sm_80a` | **`sm_80` / `sm_90a` / `sm_120a`** | Enables Ampere `cp.async` + warp MMA |
| `cake_tinygemm2_tcgen05` B200 `4×2880×2880` latency | `8.29 us` | **`7.26 us`** | **`1.1419x` speedup** (`12` programs/arch, `36` total) |
| `cake_batch_deepgemm_fp8` `K=7168` 2-CTA M-blocks & bad elements | `n_l2_group=1` (`[0, 1]`, `97,355` bad) | **`n_l2_group=8` (`[0, 0]`, `0` bad)** | Fixes >32-row group corruption on B200/GB300 |
| `cake_blackwell_softmax` 41-shape geomean speedup (B200 / GB300) | `1.000x` | **`1.283x` / `1.263x`** | `1×32000`: `3.26 -> 2.53 us` (`1.291x`) |
| CUTLASS block-scale epilogue Group-3 scaling error (`ad9bd53bda`) | `37.5` (`12.5` vs `50.0`) | **`0.0` (`[12.5, 25.0, 37.5, 50.0]`)** | Preserves per-group norm constant index |
| TileLang Ascend PTO 4-lane `Ramp` emission (`lane=2`) | 4 positional args (compile error) | **`[31, 34, 37, 40]` typed vector** | Validated via `vector_from_list` |
