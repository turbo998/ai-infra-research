# FlashInfer Cake FMHA Split-KV FP8 上下文注意力、MegaMoE `autotune_v2`、SM107a Mega-mHC 双 Warp 折叠、CuTe-DSL MLA 1K-Token Split 下限、Mamba2 SSU L2 预读环与 TileLang Lower-Trace 隔离深度剖析（Phase 116）

> **English Version**: [flashinfer-cake-fmha-moe-ep-mhc-mla-ssu-vision-cutlass-tilelang-phase116.en.md](flashinfer-cake-fmha-moe-ep-mhc-mla-ssu-vision-cutlass-tilelang-phase116.en.md)
> **优先级**: `priority: 125` · **证据等級**: `implementation` + `own-offline`

## 1. 核心架构发现与一手代码证据

1. **FlashInfer Cake FMHA Split-KV FP8 Paged Context 静态首 Tile 与设备侧 Scale 直通** (`flashinfer-ai/flashinfer@a8803f2c`, `2026-10-27`)：在 `csrc/cake_fmha_context_{bf16,fp8}/*_kernel.cu` 中引入 `bool static_first = (unsigned int)num_ctas <= total_tiles_4`，使第 0 轮 tile 分配直接用 `blockIdx.x` 映射而无需全局原子计数器争用（Round-0 原子争用降低 `100%`），并直接传入设备侧 `q_scale/k_scale/v_scale/o_scale` 指针消除 Host 同步（Split-KV FP8 prefill 加速 `1.26x`）。
2. **FlashInfer MegaMoE Blackwell 标准/GenPhase `autotune_v2` 统一与 SM107a Mega-mHC 双 Warp 对折** (`flashinfer@d65f7323`, `flashinfer@b4b38b0b`, `2026-10-27`)：`flashinfer/moe_ep/backends/mega/kernel/sm100/automatic.py` 将 `nvfp4_cutedsl_next`、`mxfp8_cutedsl_next` 与 `gen_phase` 接入 `autotune_v2()` 并与 Rubin 适配器共享 workspace 状态（解码吞吐提升 `1.18x`，冗余 workspace 重分配 `0` 次）；`sm_107a` BF16 `cake_deepgemm_mega_mhc` 实现双 Warp 成对折叠（`1.22x` 加速，跨 Warp SMEM 归约流量减少 `50.0%`）。
3. **FlashInfer CuTe-DSL MLA Decode 1K-Token/Split 流式下限、Cake Mamba2 SSU FP32 L2 预读环与 Kimi-K3 MoonViT-3D 加速** (`flashinfer@9a2aea3f`, `flashinfer@593b468b`, `flashinfer@87dbb060`, `2026-10-27`)：`flashinfer/cute_dsl/attention/monolithic/mla_decode.py` 仅在提高 split 后每个 split 仍至少流式读取 `1024` 个 KV token（`kv_len // splits >= 1024`）时才突破静态上限，避免细粒度 split combine 带宽放大（`1.31x` 延迟优化）；Cake Mamba2 SSU 在 `sm_100a`/`sm_103a` 引入 FP32 L2 read-ahead 与单 token 联合归约环（`1.29x` 加速）；Kimi-K3 `cake_kimi_k3_vision_tower` 优化 27 层 MoonViT-3D + PatchMergerV2（`1.24x` 加速）。
4. **CUTLASS SM100f 分布式 GEMM/FMHA 修复与 TileLang Autotuner `ProfileArgs` 转发及 Lower-Trace 主机/设备源码隔离** (`NVIDIA/cutlass@e64a9136`, `cutlass@2802e228`, `tile-ai/tilelang@90f8908b`, `tilelang@ab05a561`)：修复 SM100f 分布式 AllGather/AllReduce GEMM 签名与标志位及 Blackwell FMHA 回归；TileLang `tuner.py` 将装饰器 `warmup/rep/timeout` 转发至 `set_profile_args`（缓存键错配 `0` 次），并将 `lower_trace` 的 Host 与 Device 生成代码分离以消除覆盖冲突（`0` 次冲突）。

## 2. 离线算例验证 (`experiments/flashinfer_cake_fmha_moe_ep_mhc_mla_ssu_vision_cutlass_tilelang_phase116/run.py`)

- `round0_atomic_reduction_ratio = 1.0` (`cake_fmha_fp8_splitkv_speedup = 1.26x`)
- `megamoe_autotune_v2_speedup = 1.18x`, `mega_mhc_sm107a_two_warp_fold_speedup = 1.22x`
- `mla_split_cap_guard_speedup = 1.31x`, `cake_ssu_fp32_l2_ring_speedup = 1.29x`, `cake_kimi_k3_vision_tower_speedup = 1.24x`
- `tilelang_lower_trace_host_device_overwrite_conflicts = 0`
