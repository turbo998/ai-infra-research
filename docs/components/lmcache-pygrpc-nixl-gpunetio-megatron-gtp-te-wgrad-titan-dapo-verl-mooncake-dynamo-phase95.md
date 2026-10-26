# LMCache pygrpc-native 服务模式、NIXL GPUNetIO 独立 QP 推进与立即释放锁、Megatron GTP 全图捕获 Scratch 生命周期、TE 分布式权重 FP32 wgrad、TorchTitan Qwen3-30B-A3B DAPO MXFP8 RL、verl HDO 恢复预加载与 QAT 规范、Mooncake 租户命名空间切换及 Dynamo 引擎自调优深挖 (Phase 95 / Priority 104)

> **证据分类 (Evidence Class)**：上游源码实现 (`implementation`) + 作者基准报告 (`author-benchmark`, TorchTitan PR #5598) + 本项目 Python 3.12 标准库离线确定性模型 (`own-offline`)。

## 1. 核心架构问题与一手源码落点

| 信源 ID | 仓库与提交 | 关键实现路径与架构不变量 |
|---|---|---|
| `lmcache-mp-pygrpc-native-server-mode-pr6031-2026` | `LMCache/LMCache` @ `a873f152f8` (PR #6031) | LMCache 多进程（MP）服务器新增 `--grpc-server-mode native`（基于 `pygrpc-native`）：由 Rust `tokio` 线程接收网络请求，经单个 Python 调度线程分发至按 handler 类型选择的线程池，消除 `sync` 模式下每个在途 RPC 独占阻塞一个 Python gRPC 线程（64 并发下从 64 个阻塞线程降至 1 个分发线程）的开销。 |
| `lmcache-bench-request-owned-blocks-verify-pr5827-2026` | `LMCache/LMCache` @ `6b2b754654` (PR #5827) | `lmcache bench server` 引入不可变的 `RequestBlocks` / `GroupBlocks` 请求独占块分配器（跨全部 TP worker 共享逻辑顺序）并在本地 worker 缓冲区内执行逐 chunk 校验和验证，消除跨请求块别名冲突与 HTTP 校验开销。 |
| `nixl-gpunetio-independent-qp-cpu-completion-pr2222-2026` | `ai-dynamo/nixl` @ `9e7800fc16` (PR #2222) | NIXL DOCA GPUNetIO 插件重构为基于 `docaSlotDesc` 环的独立 QP 推进与 CPU 本地完成度报告，并在 CUDA 核函数中获取 SDK GPU 锁完成 SQ 槽位预留（约 `0.15 us`）后**立即释放锁**，再执行 WQE 准备与提交（约 `1.85 us`），将多流/多 QP 并发下的 GPU 锁持有时间缩短 `13.33x`（`2.00 us -> 0.15 us`），通过 DOCA 3.5 全部 27 项硬件认证用例。 |
| `megatron-gtp-preserve-scratch-lifetime-capture-pr7780-2026` | `NVIDIA/Megatron-LM` @ `ed9c2c8c1c` (PR #7780) | 修复 Generalized Tensor Parallelism (GTP) 在全迭代 CUDA Graph (`_FULL_ITERATION`) 捕获期间对称内存 scratch 缓冲区被提前复用的问题：在 `classify_gtp_chains` 中为每条链显式创建 `ag_event`、`_recompute_ag_event` 与 `rs_event` 3 个内部事件向图内存池暴露拓扑序，并在捕获前调用 `symmetric_wgrad_pool.reset_completed_reuse_events()`。 |
| `te-grouped-wgrad-main-grad-dtype-dist-weights-pr3397-2026` | `NVIDIA/TransformerEngine` @ `e32e1238b0` (PR #3397) | 修复分布式权重（Distributed Weight）在 Grouped Linear / Grouped MLP 反向传播中的 `wgrad` 精度截断：新增 `weight_grad_dtype(w, compute_dtype)` 从 `w.main_grad.dtype`（FP32）而非 `compute_dtype`（BF16）推导 `wgrad` 类型，使路径变为 `wgrad alloc fp32 -> reduce fp32 -> accum fp32`，避免每秩小梯度信号（如 `1.0005` 在 BF16 下被截断为 `1.0`，8 秩损失 `0.004`）在 reduce-scatter 前丢失。 |
| `torchtitan-qwen3-30b-a3b-dapo-math-rl-config-pr5598-2026` | `pytorch/torchtitan` @ `df4e4dc22a` (PR #5598) | 新增 Qwen3-30B-A3B DAPO 数学 8K RL 训练配置（4×B200 Trainer + 4×B200 Generator，DP4/TP1/EP4）：启用 `mxfp8_mega_fused_inference=True` 后，作者实测单步耗时从 `24.3 s` 降至 `22.7 s`（`1.07x`），等待 Rollout 耗时从 `15.8 s` 降至 `13.6 s`（`1.16x`），单请求生成 decode 耗时从 `20.2 s` 降至 `17.8 s`（`1.14x`）。 |
| `verl-megatron-onload-optimizer-before-ckpt-resume-pr8235-2026` | `volcengine/verl` @ `0bd668c1aa` (PR #8235) | 修复 `verl` Megatron 引擎同时开启 `optimizer_offload=True` 与 `optimizer_cpu_offload=True`（`HybridDeviceOptimizer`, HDO）时的断点恢复崩溃：在 `load_checkpoint` 调用 `load_state_dict` 前先执行优化器 onload，避免 HDO 按 `is_cuda` 将停留在主机内存的 FP32 分片误判为纯 CPU 参数并在首个 `optimizer.step()` 触发 `KeyError` 或 `cpu`/`cuda` 设备失配。 |
| `verl-megatron-preserve-bridge-specs-causal-mask-qat-pr8361-2026` | `volcengine/verl` @ `4f8f090af3` (PR #8361) | 修复 Megatron QAT 强行用 `modelopt_transformer_layer_spec` 覆盖 20+ 个自定义 Megatron-Bridge 模型（OLMoE、Qwen3-Next、DeepSeek-V2/V3/V4、Kimi、MiniMax 等）导致丢层/权重形状不匹配的问题，并为 `CompressedTensorsW4A8Mxfp4MoEMethod` 补齐 `activation_key = kFp8Dynamic128Sym`。 |
| `mooncake-master-tenant-namespaces-evictor-quota-pr4887-2026` | `kvcache-ai/Mooncake` @ `45bbc5d16b` (PR #4887) | Mooncake Store `MasterService` 彻底移除旧的 1024 分片 `MetadataShard` 锁架构，全面切换至 `metadata::NamespaceTable`（每租户独立路由）与 `ObjectRoute` 逐键锁（`MetadataAccessorRW/RO` 仅通过路由守卫锁定目标 key），使同分片不相关 key 的锁竞争降为 `0`。 |
| `dynamo-planner-engine-owned-retune-replay-router-pr17095-2026` | `ai-dynamo/dynamo` @ `dea508e1c1` (PR #17095) | Dynamo Planner 新增引擎自持重调优（engine-owned retune）与回放路由器控制（`replay_adapter.py`），并通过 `dynamo.aisimulate.domains.expand_domain` 统一展开离散范围与枚举参数空间。 |

## 2. 可复现实验

- **实验命令**：`python experiments/lmcache_pygrpc_nixl_gpunetio_megatron_gtp_te_wgrad_titan_dapo_verl_mooncake_dynamo_phase95/run.py --check`
