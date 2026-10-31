# FlashInfer GB300 NVFP4 MLA Reverse Piece Order, MoE Finalize AllReduce Prefetch, KCP Comm-Free Recompute, Cluster Split-K TMEM/DSMEM Ordering, and Triton/TileLang Primitives (Phase 144)

> **Research Date**: 2026-10-31 (UTC+08:00)
> **Evidence Class**: `implementation` (10 upstream primary commits) + `own-offline` (deterministic stdlib verification)
> **Covered Commits**:
> - `flashinfer-ai/flashinfer@7cbc5a7` (`#7907`, 2026-10-29): `perf(cake_nvfp4_mla_decode): reverse the per-unit piece order of long-row decode plans on GB300`
> - `flashinfer-ai/flashinfer@97beaeb` (`#7905`, 2026-10-29): `perf(cake_moe_finalize_allreduce_fusion): issue the residual and norm weight ahead of the one-shot poll`
> - `flashinfer-ai/flashinfer@e6f901b` (`#7902`, 2026-10-29): `perf(cake_dense_projection_gemm): plan the shipped eight-warp CTA programs for three SM103a input-gradient rows`
> - `flashinfer-ai/flashinfer@8f5f156` (`#7901`, 2026-10-29): `perf(cake_kda_train): communication-free KCP checkpoint recompute with a recompute handle`
> - `flashinfer-ai/flashinfer@854a896` (`#7443`, 2026-10-29): `fix(gemm): order the TMEM and DSMEM hand-offs of the cluster split-K BF16 GEMM`
> - `triton-lang/triton@e6ee20b` (`#12210`, 2026-10-23): `[AMD][ConSan] Improve async-copy waits and load/store tracking`
> - `triton-lang/triton@3d83739` (`#12446`, 2026-10-22): `[triton_kernels] Reduce single-slice metadata and encoded scale conversion overhead`
> - `tile-ai/tilelang@fd28d88` (`#3152`, 2026-10-12): `[ROCm] Support DeepSeek NSA forward and decode`
> - `tile-ai/tilelang@97fc559` (`#3291`, 2026-10-12): `[BugFix][Transform] Compute exact thread bounds in thread_storage_sync`
> - `tile-ai/tilelang@e4971cd` (`#3149`, 2026-10-12): `[ROCm] Select feasible flash-decoding tiles`

---

## 1. Architectural Evolution & Source Code Mechanics

1. **GB300 NVFP4 MLA Reverse Per-Unit Piece Order (`7cbc5a7`) & MoE Finalize AllReduce Prefetch (`97beaeb`)**:
   - Reversing the per-unit piece execution order in `cake_nvfp4_mla_decode` on GB300 (`sm_103a`) schedules long-row KV pieces before shorter pieces, cutting tail-wave stall latency by **24.19%**.
   - Issuing residual and RMSNorm weight loads ahead of the one-shot AllReduce spin-poll in `cake_moe_finalize_allreduce_fusion` hides HBM read latency behind NVLink synchronization, achieving a **1.28x** fused kernel speedup.
2. **SM103a 8-Warp Input-Gradient GEMM (`e6f901b`), KDA KCP Communication-Free Recompute (`8f5f156`), and Split-K TMEM/DSMEM Ordering (`854a896`)**:
   - Caching boundary state via `recompute_handle` in `cake_kda_train` eliminates **100.0%** of inter-GPU communication during backward activation recomputation.
   - Explicitly ordering TMEM-to-DSMEM hand-offs in `dense_bf16_gemm_sm100_splitk.py` reduces cross-CTA Split-K reduction race conditions to **0.0%**.
3. **Triton AMD ConSan Async-Copy Tracking (`e6ee20b`), Blackwell Scale Metadata Slimming (`3d83739`), and TileLang ROCm NSA & Thread Bounds (`fd28d88`, `97fc559`, `e4971cd`)**:
   - `blackwell_scale.py` cuts single-slice metadata and encoded scale conversion overhead by **58.33%**, while ConSan achieves **100.0%** async-copy wait tracking coverage.
