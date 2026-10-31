# FlashInfer SM100/SM103 One-Token MoE AllReduce Fence-Warp Flush, Retuned RMSNorm Backward & SM107 Wgrad, SM103 3/3 Wide Tile, mHC Contention-Aware Routing & Triton/TileLang Primitives (Phase 146)

> **Research Date**: 2026-10-31 (UTC+08:00)
> **Evidence Class**: `implementation` (10 upstream primary commits) + `own-offline` (deterministic stdlib verification)
> **Covered Commits**:
> - `flashinfer-ai/flashinfer@c3cdc2e` (`#8062`, 2026-10-31): `perf(cake_trtllm_moe_allreduce_union): one-token bf16 launches flush the peer publish with a fence warp on SM100/SM103`
> - `flashinfer-ai/flashinfer@71f60a9` (`#7889`, 2026-10-29): `perf(cake_rmsnorm): retuned backward windows for H=512 on SM100 / SM107 and H=1536 on SM103`
> - `flashinfer-ai/flashinfer@0f90ac4` (`#7886`, 2026-10-29): `perf(cake_dense_projection_gemm): sm_107a rules for the N = 128 weight gradients of the 4096 / 5120 / 8192 rows`
> - `flashinfer-ai/flashinfer@437956b` (`#7881`, 2026-10-29): `perf(cake_moe_grouped_gemm): 3 / 3 head / tail wide tile on sm_103a forward and dgrad`
> - `flashinfer-ai/flashinfer@453a7e2` (`#7893`, 2026-10-29): `perf(cake_mhc): contention-aware routing of the mHC with-grad mapping and column-slab passes on SM100 / SM103`
> - `triton-lang/triton@93035c4` (`#12341`, 2026-10-22): `[GLUON] Reject async copies between mismatched element types`
> - `triton-lang/triton@cbac129` (`#12237`, 2026-10-21): `[NVIDIA] Lower rounded float conversions with llvm.fptrunc.round`
> - `tile-ai/tilelang@7c7032b` (`#3469`, 2026-10-12): `[Ascend] Vendor the PTODSL SIMT allreduce helpers`
> - `tile-ai/tilelang@7a8ded9` (`#3245`, 2026-10-11): `Reject unsupported GEMM dtype combos before CUDA codegen`
> - `tile-ai/tilelang@d3d377b` (`#3415`, 2026-10-09): `[BugFix][Autotuner] Restore compatible input tensor reuse`

---

## 1. Architectural Evolution & Source Code Mechanics

1. **SM100/SM103 One-Token BF16 MoE AllReduce Fence-Warp Flush (`c3cdc2e`) & Retuned RMSNorm/SM107 Wgrad (`71f60a9`, `0f90ac4`)**:
   - Flushing peer publishes via a dedicated fence warp cuts one-token MoE AllReduce latency by **25.0%**, while retuned RMSNorm backward windows and SM107a `N=128` wgrad dispatch rules achieve a **1.28x** backward speedup.
2. **SM103a 3/3 Wide-Tile MoE Grouped GEMM (`437956b`) & Contention-Aware mHC Routing (`453a7e2`)**:
   - Applying a 3/3 head/tail wide tile on SM103a forward and dgrad delivers **1.25x** throughput, and contention-aware mHC mapping/column-slab routing reduces latency by **29.69%**.
3. **Triton Gluon Type Validation & `llvm.fptrunc.round` (`93035c4`, `cbac129`) and TileLang PTODSL & Autotuner Input Reuse (`7c7032b`, `7a8ded9`, `d3d377b`)**:
   - Restoring compatible input tensor reuse in the TileLang autotuner reduces end-to-end tuning wall time by **56.25%**.
