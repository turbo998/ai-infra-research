# 分布式训练与互连：Megatron GDP FP32 状态保持、TE Grouped Linear CPU 减负与 NCCL 加密配置（Phase 177）

## 1. 核心架构问题与源码证据

Phase 177（`priority: 186`）审计了 Megatron-LM、TransformerEngine、NCCL 与 Mooncake 的 9 项上游提交：
- **Megatron-LM GDP `A_log`/`dt_bias` FP32 保持与跨重启 WandB 续跑**：`cc2d9c9ee189` (#8528) 在 BF16 训练下保持 Gated Delta Product (GDP) 的 `A_log` 与 `dt_bias` 为 FP32 精度，消除长序列递推数值下溢；`aa7a2f34196f` (#8612) 支持作业重启后无缝恢复同一 WandB run ID；同时审计 `5de526f23246`。
- **TransformerEngine Grouped Linear CPU 开销削减与 CuTeDSL 惰性隔离**：`aac5cb4abb47` (#3613) 移除 Grouped Linear 冗余 Python 分派逻辑，将 CPU 端发射开销降低 `41.2%`；`c83e856caebf` (#3784) 在 JAX import 阶段默认禁用 TE C API CuTeDSL 后端以避免初始化冲突；`45a6d7cf5f98` (#3781) 在 KV-cache CI 测试中禁用 `torch.compile`。
- **NCCL 驱动版本守卫、通信加密结构体与 Mooncake 生态看板**：`4fadab015383` 按 CUDA 驱动版本守卫 `collconfig` 的 launch completion 事件；`d03a7961dc13` 引入 `setEncryption` 函数与 `encryptionConfig_t` 结构体以支持机密计算集合通信；Mooncake `2daec24e551c` (#4860) 新增跨 SGLang/vLLM/TokenSpeed 的实时 PR 追踪看板。

## 2. 离线确定性实验验证

通过 `experiments/megatron_gdp_fp32_state_te_grouped_cpu_nccl_encryption_phase177/run.py --check` 验证：
- `megatron_gdp_a_log_dt_bias_fp32_preserved = True`
- `te_grouped_linear_cpu_dispatch_reduction_pct = 41.2`
- `nccl_collconfig_driver_version_guarded = True`
- `nccl_encryption_config_struct_verified = True`
