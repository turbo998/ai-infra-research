# FA4 FwdConfig 与 SM100 hd256 反向调度、veRL THD 对齐与原始对数概率监控、Megatron GDP/分位数均衡、Mooncake VMM/PID/TENT RDMA、NIXL、Dynamo、TE 与 DeepEP V2.5

> English version: [fa4-fwd-config-sm100-hd256-bwd-verl-thd-logprob-megatron-gdp-qb-mooncake-nixl-dynamo-te-deepep.en.md](fa4-fwd-config-sm100-hd256-bwd-verl-thd-logprob-megatron-gdp-qb-mooncake-nixl-dynamo-te-deepep.en.md)
> 证据等级：`implementation`（上游第一手源码提交）+ `own-offline`（本项目 Python 3.12 标准库确定性仿真器，非本项目 GPU/RDMA 硬件实测）

## 1. 核心架构问题与源码演进路径

Phase 84（`priority: 93`）针对训练、注意力内核与 KV/RDMA 传输栈（`Dao-AILab/flash-attention`、`volcengine/verl`、`NVIDIA/Megatron-LM`、`kvcache-ai/Mooncake`、`ai-dynamo/nixl`、`ai-dynamo/dynamo`、`NVIDIA/TransformerEngine`、`deepseek-ai/DeepEP`）的 13 个第一手提交进行了源码审计：

1. **FlashAttention-4 不可变 `FwdConfig`、SM120 自动 `num_splits` 与 SM100 `hd256` 反向 2-CTA 调度（[#2732](https://github.com/Dao-AILab/flash-attention/commit/e7a6402a9ff099de5eb628ce207a88a3f0aa5378)、[#2758](https://github.com/Dao-AILab/flash-attention/commit/1fc0221183e799363001394a12dd9e78e15f5dda)、[#2964](https://github.com/Dao-AILab/flash-attention/commit/76d9331028192d06da018c034979765aa0f22899)）**：
   - 新增纯 Host 侧 `FwdConfig` 选择与校验器（不读取张量内容、不同步 CUDA、不分配设备内存），并在 `flash_fwd_sm100.py` 中将 KV Stage 上限封顶为 `32`（防止 `hd_padded=16` 时无界公式算出 52 Stages 溢出 `227 KB` SMEM 上限），同时对 `hd=(192, 128)` 采用非对称 SMEM 容纳 3 Stages。
   - SM120 前向支持传入自动 `num_splits <= 0` 并收敛为 `num_splits = 1`；SM100 `hd256` 反向统一引入枚举 `CtaSchedule`（`ONE_CTA`、`TWO_CTA_PIPELINED`、`TWO_CTA_SERIAL`）以显式编排 5 个反向 GEMM。
2. **veRL 配方无关 THD `quant_align_size` 与 Rollout/Actor 原始对数概率 1-ULP 偏差检测（[#8325](https://github.com/verl-project/verl/commit/b1a98e1d425e44180ddbf8d87aa074dde71c5466)、[#8025](https://github.com/verl-project/verl/commit/16ec093cc4a235c603c29aed1f2c4e9f551fab48)）**：
   - 将 `preprocess_packed_seqs` 与 `preprocess_thd_engine` 的 `use_fp8_padding`/`fp8_recipe` 统一抽象为整数 `quant_align_size=16`（配合 `min_local_rows=64, cp_layout="dualpipev"`），并同步用于 Top-K 蒸馏张量填充。
   - 揭示了仅监控概率差 `|exp(lp_rollout) - exp(lp_actor)|` 的盲区：当 `rollout = -0.10000000149` 与 `actor = -0.10000000894`（相差 1 ULP）时，`torch.exp()` 在 FP32 下舍入为完全相同的值（`probs_diff == 0`）。新增 `training/rollout_logprobs_mismatch_count`（窗口内求和而非求均值）与 `rollout_logprobs_diff_max/mean` 直接捕获原始 Log-Prob 漂移。
3. **Megatron-LM 融合 `gdp_prefill_prepare` 与全局批次分位数专家偏置均衡（[#8172](https://github.com/NVIDIA/Megatron-LM/commit/b291235082b867187c6a773dcac6512785828163)、[#7394](https://github.com/NVIDIA/Megatron-LM/commit/92641a421f40eeb40fc83f3f13ec83ca652c8b13)）**：
   - 将 Gated Delta Product (GDP) 的 Prefill 预处理融合进 `gdp_prefill_prepare` 内核入口；在 MoE 路由中集成 `get_updated_expert_bias_with_quantile` 与 TE `fused_topk_with_score_function` 的 `qb_histogram` / `qb_bin_bounds` / `qb_histogram_mode` 全局批次分位数均衡。
4. **Mooncake NVLink IPC VMM/PID 命名空间诊断与 TENT RDMA 通知槽位覆写修复（[#4873](https://github.com/kvcache-ai/Mooncake/commit/b66bb2b5f650392a2a65aa4f4dcd816808fb44e9)、[#4434](https://github.com/kvcache-ai/Mooncake/commit/dae22d472e31f6aac92ee1dfe938dc1441b58850)）**：
   - `nvlink_intra` 在 `cudaIpcGetMemHandle` 失败时通过 `cuMemRetainAllocationHandle` 精准识别 `cuMemCreate` 虚拟内存分配（如 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`），并在 `cudaIpcOpenMemHandle` 返回 `cudaErrorDeviceUninitialized` 时提示容器间 PID 命名空间冲突（需 `--ipc=host --pid=host`）。
   - TENT `RdmaEndPoint::sendNotification()` 修复了严重的数据损坏隐患：旧代码在 `ibv_post_send()` 前执行 `wr.wr_id = notify_send_wr_id_++`；当存在 255 个在途 Send（`kNotifyMaxPendingSends = 256`）且一次 Post 失败时，`notify_send_wr_id_` 被推进但未计入 `notify_pending_count_`，导致下一次 Send 分配到槽位 `(255 + 1) % 256 = 0`，直接覆写仍在 DMA 传输中的最老在途缓冲区！修复后仅在 `ibv_post_send` 成功后递增 `notify_send_wr_id_`。
5. **NIXL 后端阶段追踪、Dynamo 原子计数器批处理、TE Triton 容量截断与 DeepEP V2.5（[#2483](https://github.com/ai-dynamo/nixl/commit/0b146a82d8efb52f40b1da872bb271978f510e94)、[#17067](https://github.com/ai-dynamo/dynamo/commit/906e89d3a304e1a329be716846304bb339571272)、[#3347](https://github.com/NVIDIA/TransformerEngine/commit/c9c92255465b4518380561b1bbb0aebb177c0b6d)、[#763](https://github.com/deepseek-ai/DeepEP/commit/def865146e95aa9ea223e277c0a8a4c6b81e5ac7)）**：
   - NIXL 在 `nixlBackendOptionalArgs` 中透传 `traceRequestId` 与 `traceExecution`；Dynamo HTTP 服务每请求累积 `pending_output_tokens` 并每 `LOCAL_FLUSH_TOKENS = 64` 刷新一次全局模型计数器，将跨核缓存行乒乓降低 **64x**；TransformerEngine Triton `_row_id_map_pass_2_kernel` 增加 `tl.where(row_id >= num_out_tokens, -1, row_id)` 防止超容量路由越界写；DeepEP V2.5 引入 EP / Bucket / Pipeline-Parallel / Engram 四类共享生命周期缓冲区与 DeepJIT 运行时内核编译。

## 2. 可复现实验与架构师练习

- 运行命令：`python experiments/fa4_fwd_config_sm100_hd256_bwd_verl_thd_logprob_megatron_gdp_qb_mooncake_nixl_dynamo_te_deepep/run.py`
- 关联证据 ID：`ev-fa4-fwd-config-sm100-hd256-bwd-verl-thd-logprob-megatron-gdp-qb-mooncake-nixl-dynamo-te-deepep-2026`
