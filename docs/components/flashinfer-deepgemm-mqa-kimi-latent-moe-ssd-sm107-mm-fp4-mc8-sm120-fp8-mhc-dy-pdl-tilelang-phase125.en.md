# FlashInfer DeepGEMM MQA, Kimi-K3 LatentMoE, SM107 SSD 1-CTA Cluster, GB300 Cluster-of-8 FP4, SM120 FP8 Weight-First, mHC Backward dY Release, PDL Barrier & TileLang Deep Dive (Phase 125 / Priority 134)

> **Evidence Classification**: Upstream source implementation (`implementation`) paired with deterministic Python 3.12 offline modeling (`own-offline`). Offline analytical models must not be conflated with on-GPU hardware profiling (`own-hardware`).

---

## 1. Primary Source Provenance & Problem Framing

1. **Cake DeepGEMM Dense MQA SM100 Partial-Block Reuse** (`flashinfer-ai/flashinfer@512db9b35193e567c21f545b19027ab68376f8d2`, PR `#7741`, 2026-10-28): Serves the `fp8:h64:full:qle2048:kvsplit` route on `sm_100a` using the `fp8:h64:partial` logits program, removing 1,396 lines of duplicate generated CUDA code (`cake_deepgemm_dense_mqa_084957a7d4ca249b561d_{kernel,binding}.cu`).
2. **Cake Kimi-K3 Stable LatentMoE L2 Prefetch & Resident-B Norm** (`flashinfer-ai/flashinfer@4158fddf0cda7c46b2da19eb0d03736d90307b11`, PR `#7742`, 2026-10-28): Adds `partial_prefetch=True` and `sb_norm` resident-B arithmetic to Kimi-K3 LatentMoE decode tails ($T \le 128$, TP1/TP8) on `sm_100a` and `sm_103a`.
3. **Cake SSD Combined SM107a One-CTA-Cluster Launches & Fast F32 Claim** (`flashinfer-ai/flashinfer@ecd4d71d893e948a68c09ae6b175a8a931cac525`, PR `#7740`, 2026-10-28): Annotates `__cluster_dims__(1,1,1)` across 56 Mamba2 SSD combined prefill programs on Rubin (`sm_107a`), adopts one-CTA tail pairs, and accelerates the F32 claim program.
4. **Cake NVFP4 Per-Token GB300 Cluster-of-8 Multicast GEMM (`mc42`)** (`flashinfer-ai/flashinfer@c9fc6cab4066ee4473df640b7ba50e540794340b`, PR `#7739`, 2026-10-28): Introduces 8-CTA cluster multicast (`_sfr_mc42`) over $2 \times 2$ pair tiles for $M \in [1793, 2048]$ on `sm_103a`, halving HBM operand and scale loads across paired CTAs.
5. **SM120 Per-Tensor FP8 Weight-First Grouped TC Schedule** (`flashinfer-ai/flashinfer@1e7a60d2320d565b8b37f741127848228b54a6e5`, PR `#7520`, 2026-10-28): Overlaps weight TMA loads with software activation gathering (`weight_first=True`, cache key `fp8-grouped-tc-w8-weight-first-v6`).
6. **Cake mHC Backward Per-Connection `dY` Release** (`flashinfer-ai/flashinfer@1fb63709b8bc12629ce06626e19f0bdc54e85024`, PR `#7718`, 2026-10-28): Passes `dY` (`508.2 MiB` at $T=16263$) and `grad_stats` along autograd edges from the finish node to the prepare node rather than retaining them on `_MHCTrainState` until step end.
7. **MoE EP Pre-Dispatch Quantization, SM90a GDP Varlen, Post-PDL-Wait Scale Reads & Prims-TS Rubin Locality Domains** (`flashinfer-ai/flashinfer@b1cf76608bb6519b5e5fc7972c490fb96c9258d4` PR `#7426`, `f24f6e820893476841fec030d6a85041dd897225` PR `#7713`, `870712281338bb2baeb94f30912f3abfaf67a920` PR `#6322`, `8ab3a49654c66ffced500309ed95377201bb68bb` PR `#6832`) and **TileLang Fixes** (`tile-ai/tilelang@714cbf150d90412ca390eeded575dd1c962d960b` PR `#3374`, `fca340e5c2ce6263555ae5fdffe5c6fa90f0735b` PR `#3367`, `dbabf135f96209f8051042db1440bcbad41a6e68` PR `#3341`, `a366af801a88aa539418d45844b8790a50f6f65c` PR `#3366`).

---

## 2. Deterministic Offline Verification (`own-offline`)

Run command:
```bash
python experiments/flashinfer_deepgemm_mqa_kimi_latent_moe_ssd_sm107_mm_fp4_mc8_sm120_fp8_mhc_dy_pdl_tilelang_phase125/run.py --check
```

| Subsystem / Metric | Baseline | Phase 125 Optimized | Delta / Architectural Impact |
|---|---:|---:|---:|
| DeepGEMM Dense MQA SM100 (`q<=2048` kvsplit) | `33.22 μs` | `27.05 μs` | **1.227x** mean speedup, **-1,396** LOC |
| Kimi-K3 LatentMoE Tail Decode ($T=128$) | `61.92 μs` | `43.28 μs` | **1.410x** peak speedup via L2 `partial_prefetch` |
| GB300 `cake_mm_fp4` Cluster-of-8 (`mc42`) | `66.81 μs` | `49.11 μs` | **1.363x** mean speedup, **-50.0%** HBM reads |
| Cake mHC Backward Retained `dY` ($L=32$) | `16,262.4 MiB` | `508.2 MiB` | **15,754.2 MiB (-96.88%)** memory saved |
| Split MoE EP NVLink Dispatch Payload | `67.11 MB` (BF16) | `18.87 MB` (NVFP4) | **-71.88%** NVLink communication volume |
| PDL `SFScale` Read Race Rate | `4.2%` | **0.0%** | **100%** race elimination after `cudaGridDependencySynchronize()` |
| TileLang Batched All-Reduce OOB Threads | `32` threads | **0** threads | **100%** shared-memory underflow elimination |
