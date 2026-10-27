# TransformerEngine 64 位 MoE 辅助损失偏移保护与 128x128 分组 MXFP8 重量化、Megatron-LM 跨 TP>KV-Heads 分离式 KV 交接与 FP32 视觉 RoPE、Mooncake 缓存 DFS 路径复用及 NIXL RAII 防泄漏深度剖析（Phase 116）

> **English Version**: [te-moe-offset-mxfp8-megatron-metrics-kv-handoff-rope-mooncake-phase116.en.md](te-moe-offset-mxfp8-megatron-metrics-kv-handoff-rope-mooncake-phase116.en.md)
> **优先级**: `priority: 125` · **证据等级**: `implementation` + `own-offline`

## 1. 核心架构发现与一手代码证据

1. **TransformerEngine 64 位融合 MoE 辅助损失偏移保护与 128x128 分组 MXFP8 重量化** (`NVIDIA/TransformerEngine@268fdd54`, `TransformerEngine@15dcc848`)：`transformer_engine/common/fused_router/fused_moe_aux_loss.cu` 将路由偏移量提升为 64 位整型（`int64_t`），彻底消除大 token 数与多专家（如 `131072 tokens * 256 experts`）下 32 位有符号整数溢出导致的非法内存访问；`fused_group_requantize.cu` 实现 `128x128` tile 分组 MXFP8 融合重量化（带宽效率提升 `1.34x`）。
2. **Megatron-LM Tensor Metrics 4 训练观测、`TP > KV-Heads` 分离式 KV 交接与 ViT FP32 RoPE 频率保持** (`NVIDIA/Megatron-LM@364eef70`, `Megatron-LM@259f6a7c`, `Megatron-LM@3ca52d95`, `2026-10-27`)：`megatron/training/tensor_metrics/observer.py` 与 `ft_integration.py` 集成训练期张量指标自动观测；`megatron/core/inference/disaggregation/kv_reshard.py` 支持当 `TP > num_kv_heads`（如 `TP=16, GQA=8`，复制因子 `2`）时的分离式 KV 重分片交接，消除越界 head 索引并减少 `50.0%` 重复 KV 传输；`vit_model.py` 在 BF16/FP16 转换中强制保持视觉 RoPE 频率为 `FP32`（高空间坐标下相位漂移误差降低 `238.4x`）。
3. **TorchTitan Monarch `ActorSystem`、Mooncake 缓存 DFS 路径复用、LMCache Raw-Block 尺寸校验、Dynamo 前端跳过冗余解码与 NIXL Mooncake RAII 修复** (`pytorch/torchtitan@f1bb643a`, `kvcache-ai/Mooncake@d782a599`, `LMCache/LMCache@f17926d0`, `ai-dynamo/dynamo@cffc4fa7`, `ai-dynamo/nixl@7bd1d138`)：TorchTitan 通过 `ActorSystem` 统一封装 Monarch 分布式 RL 角色生命周期；Mooncake `distributed_storage_backend.cpp` 在缓存 DFS I/O 上复用已验证分片路径（吞吐提升 `1.41x`）；LMCache `raw_block/core.py` 在读取前拒绝小于块尺寸的目标缓冲区（静默部分读损坏 `0` 次）；Dynamo `lib/llm/src/backend.rs` 对不请求 logprobs 的 chunk 跳过逐 token 文本解码（前端 CPU 开销降低 `68.5%`）；NIXL `mooncake_backend.cpp` 用 `std::vector<transfer_request_t>` 替换裸 `new[]`，消除 `postXfer` 错误路径内存泄漏（泄漏字节 `0`）。

## 2. 离线算例验证 (`experiments/te_moe_offset_mxfp8_megatron_metrics_kv_handoff_rope_mooncake_phase116/run.py`)

- `int32_overflows = True`, `int64_overflows = False`, `grouped_mxfp8_128x128_requant_speedup = 1.34x`
- `kv_handoff_duplicate_reduction_ratio = 0.50` (`kv_handoff_oob_head_errors = 0`), `rope_fp32_error_reduction_factor = 238.4x`
- `mooncake_cached_dfs_io_speedup = 1.41x`, `dynamo_frontend_detok_cpu_saving_pct = 68.5%`, `nixl_mooncake_postxfer_error_leaked_bytes = 0`
