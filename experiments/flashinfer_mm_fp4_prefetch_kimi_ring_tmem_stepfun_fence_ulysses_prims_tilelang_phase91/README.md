# Phase 91 Track A: FlashInfer Cake NVFP4 Prefetch, Kimi-K3 Ring & TMEM Scrub, Proxy Fences, Ulysses Direct NVLink, PrimTS MLA & TileLang Scalar ABI

Deterministic Python 3.12 standard-library model covering 10 first-hand commits:

1. `flashinfer-ai/flashinfer#7424` (`fc933c5bdd`): Entry L2 weight + block-16 scale prefetch across `CHAIN_EP_STAGES = 4` before PDL grid dependency wait (`1.010x–1.076x` on B200 across 64 rows, `1.012x–1.081x` on GB300 across 56 rows).
2. `flashinfer-ai/flashinfer#6900` (`6785a485ac`): Kimi-K3 FP8 projection resident decode slot `(res_stage_m + 3u) % 4u` eliminating 4th-tile unsigned underflow (`0xFFFFFFFF`).
3. `flashinfer-ai/flashinfer#6899` (`4bd511ac3d`): Kimi-K3 MLA wide kernel uniform `tcgen05.st` TMEM scrub address `o_tmem_row_base_1` across all 32 lanes.
4. `flashinfer-ai/flashinfer#7422` (`9cce37835f`): Faster Kimi-K3 27-layer MoonViT-3D + PatchMergerV2 vision tower on SM100/SM103 (`1.074x` on B200 `800x600`, `1.058x` on B300 `224x224`).
5. `flashinfer-ai/flashinfer#7428` (`6bd679e419`): StepFun MoE MXFP8 tile-256 FC1 `fence.proxy.async.shared::cta` between `cp.async.wait_group 1` and `tcgen05.cp` scale-factor read.
6. `flashinfer-ai/flashinfer#7437` (`05e425835b`): MiniMax-H3 SM120 CTA-epilogue stage release fence before `mbarrier.arrive`.
7. `flashinfer-ai/flashinfer#7440` (`dc50addf12`): Cake Ulysses direct-write NVLink all-to-all into collectively registered IPC outputs (`1.108x` scatter / `1.107x` gather at world 6, `1.094x` gather at world 8 on B300).
8. `flashinfer-ai/flashinfer#7170` (`396ffbad72`): KDA BF16 BT16 prepare tail-repair `fence.proxy.async.shared::cta` before next chunk's raw TMA load.
9. `flashinfer-ai/flashinfer#7026` (`6ba6bdf377`): PrimTS dense/sparse MLA decode `softmax_rescale_threshold_log2` with FP8 P scale `448 * 2**-T`.
10. `tile-ai/tilelang#3602` (`f2b8866e73`): JIT scalar output shape resolution by parameter position (`get_output_scalar_dim_map`) without appending duplicate ABI arguments.
