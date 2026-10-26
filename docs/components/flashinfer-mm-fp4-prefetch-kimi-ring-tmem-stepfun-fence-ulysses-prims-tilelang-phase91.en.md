# Phase 91 Track A: FlashInfer Cake NVFP4 L2 Weight Prefetch, Kimi-K3 Ring Slot & Wide MLA TMEM Scrub, Blackwell Async Proxy Fences, Ulysses Direct NVLink, PrimTS MLA Skip Rescale & TileLang Scalar ABI Resolution

> **中文版本**: [flashinfer-mm-fp4-prefetch-kimi-ring-tmem-stepfun-fence-ulysses-prims-tilelang-phase91.md](flashinfer-mm-fp4-prefetch-kimi-ring-tmem-stepfun-fence-ulysses-prims-tilelang-phase91.md)  
> **Phase Status**: Phase 91 (`priority: 100 — 100-Priority Milestone!`) · Question `q-flashinfer-mm-fp4-prefetch-kimi-ring-tmem-stepfun-fence-ulysses-prims-tilelang-phase91` · Claim `claim-flashinfer-mm-fp4-prefetch-kimi-ring-tmem-stepfun-fence-ulysses-prims-tilelang-phase91`  
> **Reproducible Experiment**: [`experiments/flashinfer_mm_fp4_prefetch_kimi_ring_tmem_stepfun_fence_ulysses_prims_tilelang_phase91/`](../../experiments/flashinfer_mm_fp4_prefetch_kimi_ring_tmem_stepfun_fence_ulysses_prims_tilelang_phase91/)

---

## 1. First-Hand Source Evidence Matrix (10 Upstream Commits)

| # | Repository & Commit | Date | Evidence Class | Core Architectural Mechanism |
|---|---|---|---|---|
| 1 | `flashinfer-ai/flashinfer` [`fc933c5bdd`](https://github.com/flashinfer-ai/flashinfer/commit/fc933c5bdd11292c30a87db38e44f1ae4dc74984) (`#7424`) | 2026-10-26 | `implementation` + `author-benchmark` | `cake_mm_fp4` streams weight tiles and block-16 scales for the first `CHAIN_EP_STAGES = 4` K-tiles into L2 before the PDL grid dependency wait (`1.010x–1.076x` on B200 across 64 rows, `1.012x–1.081x` on GB300 across 56 rows) |
| 2 | `flashinfer-ai/flashinfer` [`6785a485ac`](https://github.com/flashinfer-ai/flashinfer/commit/6785a485acaa6da113672da819bd99eb99e2cf18) (`#6900`) | 2026-10-26 | `implementation` | `cake_kimi_k3_fp8_projection` resident decode programs compute `(res_stage_m + 3u) % 4u`, preventing unsigned underflow (`0xFFFFFFFF`) when `res_stage_m` wraps to `0` on a 4th token tile |
| 3 | `flashinfer-ai/flashinfer` [`4bd511ac3d`](https://github.com/flashinfer-ai/flashinfer/commit/4bd511ac3dbde1851a80fa9d8f4df7315ef83750) (`#6899`) | 2026-10-26 | `implementation` + `author-benchmark` | `cake_kimi_k3_mla` 2-CTA wide kernel sets `scrub_row = o_tmem_row_base_1` uniformly across all lanes to satisfy the `tcgen05.st` lane-quarter address requirement |
| 4 | `flashinfer-ai/flashinfer` [`9cce37835f`](https://github.com/flashinfer-ai/flashinfer/commit/9cce37835fc8cb38c09409335c36c0f82b88bb6f) (`#7422`) | 2026-10-26 | `implementation` + `author-benchmark` | `cake_kimi_k3_vision_tower` (27-layer MoonViT-3D + PatchMergerV2) adds `l_e8_tma1x` / `l_e8_tma1x_h` TMA-epilogue 2-CTA tiles at `T >= 8192` (`1.011x–1.074x` on B200, `1.007x–1.058x` on B300) |
| 5 | `flashinfer-ai/flashinfer` [`6bd679e419`](https://github.com/flashinfer-ai/flashinfer/commit/6bd679e419ff52aa6eb382f84a70a59429a3b642) (`#7428`) | 2026-10-26 | `implementation` | `cake_stepfun_moe` MXFP8 tile-256 FC1 inserts `fence.proxy.async.shared::cta` between `cp.async.wait_group 1` and `tcgen05.cp` scale reads |
| 6 | `flashinfer-ai/flashinfer` [`05e425835b`](https://github.com/flashinfer-ai/flashinfer/commit/05e425835b8d9777b01972d10a2083da6b5e92aa) (`#7437`) | 2026-10-26 | `implementation` | `cake_minimax_h3` SM120 CTA-epilogue programs fence shared-memory stage release before `mbarrier.arrive` |
| 7 | `flashinfer-ai/flashinfer` [`dc50addf12`](https://github.com/flashinfer-ai/flashinfer/commit/dc50addf1244fd614fa50477179f788369221d1a) (`#7440`) | 2026-10-26 | `implementation` + `author-benchmark` | `cake_ulysses` adds 24 direct-write NVLink all-to-all programs writing straight into collectively registered IPC outputs (`1.108x` scatter at world 6, `1.094x` gather at world 8 on B300) |
| 8 | `flashinfer-ai/flashinfer` [`396ffbad72`](https://github.com/flashinfer-ai/flashinfer/commit/396ffbad72133e3d364efdf91b25f10cb4844522) (`#7170`) | 2026-10-26 | `implementation` | `cake_kda` BF16 BT16 prepare inserts `fence.proxy.async.shared::cta` after generic tail zeroing before the next chunk's raw TMA load |
| 9 | `flashinfer-ai/flashinfer` [`6ba6bdf377`](https://github.com/flashinfer-ai/flashinfer/commit/6ba6bdf377ff65a5233c7c89b55fbbf1f8e2e655) (`#7026`) | 2026-10-26 | `implementation` | PrimTS dense/sparse MLA decode adds `MLAAttentionTuning.softmax_rescale_threshold_log2` (`P_scale = 448 * 2**-T` up to `T = 8` for E4M3, `T = 32` for BF16) |
| 10 | `tile-ai/tilelang` [`f2b8866e73`](https://github.com/tile-ai/tilelang/commit/f2b8866e733f667dd273ac383fbe30fab2098d1e) (`#3602`) | 2026-10-26 | `implementation` | TileLang JIT resolves dynamic scalar output dimensions by parameter position (`get_output_scalar_dim_map`) without appending duplicate scalar arguments to Cython/PTO ABIs |

---

## 2. Implementation Deep Dive

### 2.1 Entry L2 Weight Prefetch Across PDL Grid Wait (`#7424`)
In `cake_mm_fp4`, the NVFP4 GEMM is launched back-to-back after per-token activation quantization using Programmatic Dependent Launch (PDL). While the GEMM waits at `cudaGridDependencySynchronize()` for activation quantization to finish, `#7424` uses the idle prefetch warp on 12 small-M cells per architecture to prefetch the first `CHAIN_EP_STAGES = 4` K-tiles of NVFP4 weights (`8,192 B`) and block-16 FP8 scales (`1,024 B`, `9,216 B/tile`, `36,864 B/CTA`) into L2.

### 2.2 Generic-to-Async Shared Memory Proxy Ordering (`#7428`, `#7437`, `#7170`)
Across Blackwell SM100/SM103/SM120 kernels, shared memory accesses issued via the generic proxy (`cp.async` or `st.shared`) are not automatically ordered before async proxy operations (`tcgen05.cp` or TMA `cp.async.bulk`) by thread-block barriers alone. Explicit `fence.proxy.async.shared::cta` instructions are required between `cp.async.wait_group` / generic tail stores and subsequent `tcgen05.cp` or TMA fills.

### 2.3 Registered NVLink Direct-Write Ulysses All-to-All & PrimTS MLA Skip Correction (`#7440`, `#7026`)
When outputs come from `UlyssesCommunicator.allocate_output`, symmetric CUDA IPC mappings allow every rank to scatter/gather head stripes directly into destination peer output buffers without staging copies. Meanwhile, PrimTS MLA decode exposes `softmax_rescale_threshold_log2` ($T$) with FP8 E4M3 pre-scaling $448 \times 2^{-T}$ to bound unrescaled exponentiated scores within E4M3 dynamic range.
