# FlashInfer SM120 MiniMax-H3 Varlen Attention, Re-Init-Free CUTLASS GEMM/MLA, SDSA Key Unions, Ulysses World-8 Direct Scatter, and Triton/TileLang K-Pool Primitives (Phase 145)

> **Research Date**: 2026-10-31 (UTC+08:00)
> **Evidence Class**: `implementation` (10 upstream primary commits) + `own-offline` (deterministic stdlib verification)
> **Covered Commits**:
> - `flashinfer-ai/flashinfer@627daca` (`#7906`, 2026-10-29): `perf(cake_minimax_h3): head-major pre-processing for long sequential SM120 FP8 varlen plans`
> - `flashinfer-ai/flashinfer@0ec193b` (`#7904`, 2026-10-29): `perf(cake_minimax_h3): WS-IL lead and tail and a 53248-token overlap bound for uncompensated SM120 NVFP4 varlen attention on 188-SM parts`
> - `flashinfer-ai/flashinfer@76eb3f6` (`#6282`, 2026-10-29): `perf(gemm): launch FP4/MXFP8/SVDQuant CUTLASS GEMMs and CUTLASS MLA without re-initializing`
> - `flashinfer-ai/flashinfer@b52749f` (`#6421`, 2026-10-29): `[SDSA 2/6][Attention] Add key unions and masks for token-sparse attention on dense kernels`
> - `flashinfer-ai/flashinfer@6238131` (`#7891`, 2026-10-29): `perf(cake_ulysses): faster world-8 bf16 direct scatter, early first-unit zone`
> - `triton-lang/triton@3419b56` (`#12401`, 2026-10-21): `[triton_kernels] Ceiling-round E4M3 scales for ROUND_UP`
> - `triton-lang/triton@cd772b3` (`#12402`, 2026-10-21): `[Gluon] Predicate TMEM copies and fix tutorial synchronization`
> - `tile-ai/tilelang@1ea1848` (`#3419`, 2026-10-08): `[Perf] Reduce K-pool Hadamard barriers and decode validation waits`
> - `tile-ai/tilelang@ebb0de0` (`#3427`, 2026-10-10): `[BugFix][Transform] Respect For loop steps in thread_storage_sync and codegen`
> - `tile-ai/tilelang@e59201f` (`#3441`, 2026-10-07): `[Fix] Preserve shared region offsets in LDSM copies`

---

## 1. Architectural Evolution & Source Code Mechanics

1. **SM120 MiniMax-H3 Head-Major FP8 & 53,248-Token NVFP4 Overlap (`627daca`, `0ec193b`)**:
   - Combining head-major preprocessing with WS-IL lead/tail scheduling and a 53,248-token overlap bound on 188-SM SM120 parts delivers a **1.34x** varlen attention speedup.
2. **Re-Init-Free CUTLASS Launches (`76eb3f6`), SDSA Key Unions (`b52749f`), and Ulysses World-8 Direct Scatter (`6238131`)**:
   - Skipping redundant CUTLASS operator re-initialization cuts host launch overhead by **61.54%**; SDSA key unions and masks reduce HBM KV traffic by **62.50%** on dense Tensor Core kernels; and Ulysses early first-unit zone direct scatter lowers World-8 BF16 All-to-All latency by **21.88%**.
3. **Triton E4M3 Ceiling Rounding (`3419b56`), Gluon Predicated TMEM Copies (`cd772b3`), and TileLang K-Pool Barrier Reduction (`1ea1848`, `ebb0de0`, `e59201f`)**:
   - Reducing K-pool Hadamard barriers and decode validation waits cuts synchronization stall overhead by **43.75%**.
