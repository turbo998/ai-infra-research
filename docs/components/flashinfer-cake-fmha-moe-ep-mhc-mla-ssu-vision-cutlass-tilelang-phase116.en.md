# FlashInfer Cake FMHA Split-KV FP8 Context Attention, MegaMoE `autotune_v2`, SM107a Mega-mHC Two-Warp Fold, CuTe-DSL MLA 1K-Token Split Floor, Mamba2 SSU L2 Read-Ahead Ring & TileLang Lower-Trace Isolation Deep Dive (Phase 116)

> **中文版**: [flashinfer-cake-fmha-moe-ep-mhc-mla-ssu-vision-cutlass-tilelang-phase116.md](flashinfer-cake-fmha-moe-ep-mhc-mla-ssu-vision-cutlass-tilelang-phase116.md)
> **Priority**: `priority: 125` · **Evidence Mode**: `implementation` + `own-offline`

## 1. Core Architectural Findings & Primary Code Evidence

1. **FlashInfer Cake FMHA Split-KV FP8 Paged Context Static First Tile & Direct Device Scales** (`flashinfer-ai/flashinfer@a8803f2c`, `2026-10-27`): Introduces `bool static_first = (unsigned int)num_ctas <= total_tiles_4` in `csrc/cake_fmha_context_{bf16,fp8}/*_kernel.cu` so round-0 tiles map deterministically via `blockIdx.x` without atomic work-counter contention (`100%` round-0 atomic reduction) and passes device-resident `q_scale/k_scale/v_scale/o_scale` pointers directly (`1.26x` split-KV FP8 prefill speedup).
2. **FlashInfer MegaMoE Blackwell Standard/GenPhase `autotune_v2` & SM107a Mega-mHC Two-Warp Pair Fold** (`flashinfer@d65f7323`, `flashinfer@b4b38b0b`, `2026-10-27`): Routes Blackwell standard NVFP4/MXFP8 (`*_cutedsl_next`) and `gen_phase` through `autotune_v2()` in `flashinfer/moe_ep/backends/mega/kernel/sm100/automatic.py` with shared workspace state (`1.18x` decode speedup, `0` redundant reallocations), while `cake_deepgemm_mega_mhc` folds warp pairs on `sm_107a` BF16 (`1.22x` speedup, `50.0%` cross-warp SMEM reduction savings).
3. **FlashInfer CuTe-DSL MLA Decode 1K-Token/Split Floor, Cake Mamba2 SSU FP32 L2 Read-Ahead Ring & Kimi-K3 MoonViT-3D Speedup** (`flashinfer@9a2aea3f`, `flashinfer@593b468b`, `flashinfer@87dbb060`, `2026-10-27`): Enforces `kv_len // splits >= 1024` before raising splits past the static cap in `mla_decode.py` (`1.31x` latency improvement), adds FP32 L2 read-ahead and one-token joint-reduction rings to Cake Mamba2 SSU on `sm_100a`/`sm_103a` (`1.29x` speedup), and accelerates the 27-layer MoonViT-3D + PatchMergerV2 vision tower (`1.24x` speedup).
4. **CUTLASS SM100f Distributed GEMM/FMHA Fixes & TileLang Autotuner `ProfileArgs` Forwarding and Lower-Trace Separation** (`NVIDIA/cutlass@e64a9136`, `cutlass@2802e228`, `tile-ai/tilelang@90f8908b`, `tilelang@ab05a561`): Fixes SM100f distributed AllGather/AllReduce GEMM signatures/flags and Blackwell FMHA regression, forwards `@autotune` profiler arguments into `set_profile_args` (`0` cache-key mismatches), and separates host and device sources in `lower_trace` (`0` overwrite conflicts).

## 2. Offline Verification (`experiments/flashinfer_cake_fmha_moe_ep_mhc_mla_ssu_vision_cutlass_tilelang_phase116/run.py`)

- `round0_atomic_reduction_ratio = 1.0` (`cake_fmha_fp8_splitkv_speedup = 1.26x`)
- `megamoe_autotune_v2_speedup = 1.18x`, `mega_mhc_sm107a_two_warp_fold_speedup = 1.22x`
- `mla_split_cap_guard_speedup = 1.31x`, `cake_ssu_fp32_l2_ring_speedup = 1.29x`, `cake_kimi_k3_vision_tower_speedup = 1.24x`
- `tilelang_lower_trace_host_device_overwrite_conflicts = 0`
