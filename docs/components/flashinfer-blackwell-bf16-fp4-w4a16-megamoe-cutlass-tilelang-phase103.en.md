# FlashInfer Blackwell BF16×FP4 Relaxed Cluster Arrive & 8-CTA Split-K, W4A16 MegaMoE EP Backend, Deferred Logits Reduction & CuTeDSL / TileLang Fixes (Phase 103)

- **Question ID**: `flashinfer-blackwell-bf16-fp4-w4a16-megamoe-cutlass-tilelang-phase103` (`priority: 112`, component: `gpu-software-stack`)
- **Evidence Status**: `own-offline` (First-hand upstream commit inspection + deterministic Python 3.12 stdlib model `experiments/flashinfer_blackwell_bf16_fp4_w4a16_megamoe_phase103/run.py`; upstream B200/B300/GB300 CUPTI tables marked `author-benchmark`)
- **Chinese Companion**: [`flashinfer-blackwell-bf16-fp4-w4a16-megamoe-cutlass-tilelang-phase103.md`](flashinfer-blackwell-bf16-fp4-w4a16-megamoe-cutlass-tilelang-phase103.md)

## 1. Primary Upstream Sources (Zero-Overlap)

1. **FlashInfer `535c18c714` (#7463)**: `perf(cake_blackwell_bf16_fp4): relaxed tcgen05 entry cluster arrive and a leaner 8-CTA split-K reduce-scatter cell`
2. **FlashInfer `124955a41b` (#7557)**: `feat(cake_mega_moe): BF16 × NVFP4 W4A16 MegaMoE EP backend for sm_100a/sm_103a`
3. **FlashInfer `7c87948b77` (#7515)**: `perf(cake_fused_qk_rope_fp8_append): regenerate fold, scan and page-prefetch stages (1.1-1.8 % geomean speedup)`
4. **FlashInfer `cbe168b01e` (#5196)**: `perf(sampling): defer logits block reduction`
5. **FlashInfer `f7d1cb9f55` (#7563)**: `perf(cake_kda): faster prepared BF16-input KDA prefill with FP32 state`
6. **FlashInfer `bbee0f77a0` (#7555)**: `perf(cake_gdn): faster backward prep, dqkg and adjoint-walk tape kernels, forward prep prefetch, two CTAs per SM for backward prep on SM100 and SM103`
7. **CUTLASS `6278db36ee` (#3725)**: `[CuTeDSL] fix: accept bare int as local_partition's index argument`
8. **CUTLASS `af3d4367a8` (#3639)**: `[CuTeDSL] Select CUDA 12.9 grouped GEMM codegen by CTA mode`
9. **TileLang `a765bb4bf2` (#3356)**: `[BugFix][Transform] Unroll the epilogue of a pipeline with a runtime trip count`
10. **TileLang `5434f0ac54` (#3346)**: `[BugFix] Treat an address_of pointer access as a write in ThreadSync`

---

## 2. Core Implementation Mechanics & Quantitative Analysis

### 2.1 Blackwell `cake_blackwell_bf16_fp4` Relaxed Entry Cluster Barrier & 16.0x Split-K DSMEM Reduction (`535c18c714`)

FlashInfer's `mm_bf16_fp4` multiplies BF16 activations by E2M1 FP4 weights (with E4M3 block scales and FP32 accumulation) across `blackwell-native` and `blackwell-tiled` backends (97 Cake programs). Commit `535c18c714` regenerates 9 programs with two targeted optimizations:
1. **Relaxed Entry Cluster Arrive (`barrier.cluster.arrive.relaxed.aligned;`)**: Because `fence.mbarrier_init.release.cluster;` already publishes shared-memory barriers across the cluster, replacing `barrier.cluster.arrive.release.aligned;` avoids a GPU-wide memory drain before warps enter their specialized roles, yielding **1.016x / 1.015x** speedups at `768/2112/2048` on GB300 (`8.692 → 8.556 µs`) and B200 (`9.368 → 9.232 µs`).
2. **8-CTA Split-K Reduce-Scatter Payload Cut (`7,168 B → 448 B`)**: At `M=1, N=4096, K=4096`, the two `blackwell-tiled` 8-CTA split-K reduce-scatter programs issue TMA loads before the startup cluster rendezvous and transmit only the single output row stored by each peer (`64 B/peer` instead of `1,024 B/peer`), reducing `mbarrier_expect_tx` from $7 \times 1024 = 7{,}168\text{ B}$ to $7 \times 64 = 448\text{ B}$ (**16.0x DSMEM reduction**) and speeding up GB300 / B200 by **1.060x** (`6.608 → 6.232 µs`) and **1.059x** (`6.784 → 6.406 µs`).

### 2.2 `sm103_bf16_nvfp4_bf16_cake` W4A16 MegaMoE EP Backend & Batch-Size Crossover (`124955a41b`)

`sm103_bf16_nvfp4_bf16_cake` fuses token dispatch, gated SwiGLU FC1, and FC2 (decoding NVFP4 weights to BF16 in registers for BF16 Tensor Core MMA with FP32 accumulation), chained via PDL into a counted top-8 combine kernel. Against FlashInfer's autotuned `sm100_bf16_nvfp4_bf16_cutedsl` on NVIDIA B300 (`sm_103a`):
- **Single-GPU Preset D (`H=7168, I=2048, E=256, top-8`)**: Delivers **1.5912x** speedup at `T=1` (`103.2 → 64.9 µs`) and `1.0204x–1.0558x` at `T=8..512`, while `cutedsl` wins at `T=4096` (`2968.5` vs `3178.5 µs`, ratio `0.9339x`).
- **8-GPU EP8 Preset G (`H=6144, I=2048, E=256, top-8, EP8`)**: Delivers **1.0839x** speedup at `T=4096` (`415.2 → 383.0 µs`), while `cutedsl` wins across `T=8..512` (`0.9136x–0.9377x`).

### 2.3 Deferred Logits Block Reduction (`cbe168b01e`) & CuTeDSL / TileLang Fixes (`6278db36ee`, `af3d4367a8`, `a765bb4bf2`, `5434f0ac54`)

- **Deferred Logits Reduction (`cbe168b01e`)**: Replaces per-tile `BlockReduce` across 38 vocabulary tiles (`V = 151,936`, `4,096` items/tile) with one thread-local candidate and a single final `BlockReduce` (**38.0x reduction**, `38 → 1`), resolving ties to the largest position (`151594`).
- **CuTeDSL (`6278db36ee`, `af3d4367a8`)**: Wraps bare `int` indices via `const(index, Int32)` in `local_partition` and selects vectorized RMEM + `-O2` only for 2-CTA grouped GEMM on CUDA 12.9.
- **TileLang (`a765bb4bf2`, `5434f0ac54`)**: Simplifies runtime pipeline epilogue extents `(n + s) - n` to constant `s` so `ptx_wait_group` receives an immediate constant, and records both `kRead` and `kWrite` for `address_of` shared-memory pointers so `ThreadSync` inserts the required `__syncthreads()` barrier (`0 → 1`).
