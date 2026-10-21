# TE GroupedMLP Split-Alignment 自动融合与 ONNX 0-D 缩放元数据、Megatron SchedulerConfig 与 TorchTitan Llama-3.2 32x RoPE / Attention Gym 按模块守卫、以及 NIXL EP 8x8 CPU Proxy 与 Mooncake Conductor Rust 客户端解析

> **更新日期**：2026-10-22 · **阶段**：Phase 53 (`priority: 62`) · **分类**：`distributed-training`  
> **关联实验**：[`experiments/te-grouped-mlp-align-megatron-sched-titan-rope32-nixl-proxy-mooncake/run.py`](../../experiments/te-grouped-mlp-align-megatron-sched-titan-rope32-nixl-proxy-mooncake/run.py)

---

## 1. 一手上游来源（First-Hand Sources）

| 仓库与提交 | 日期 (UTC) | 核心技术贡献 |
|---|---|---|
| [`NVIDIA/TransformerEngine@9d9bd9e5c4`](https://github.com/NVIDIA/TransformerEngine/commit/9d9bd9e5c4764633c252fdad905d8f60b6d95098) | 2026-10-15 | `[PyTorch]`：废弃 `NVTE_CUTEDSL_FUSED_GROUPED_MLP` 环境变量门控，改由 `GroupedLinear(split_alignment=256/128)` 驱动 `GroupedMLP_CuTeGEMMGLU` 自动融合（#3584） |
| [`NVIDIA/TransformerEngine@33f3b35619`](https://github.com/NVIDIA/TransformerEngine/commit/33f3b3561943fcd9dd2f544f67dc2dfd62efdc9e) | 2026-10-15 | `Fix ONNX custom-op metadata`：修复 `onnx_cs_quantize_fp8_op` 假张量缩放输出形状为 0-D `()` 并通过 `torch.library.opcheck` 验证（#3659） |
| [`NVIDIA/Megatron-LM@e4eb89b3bb`](https://github.com/NVIDIA/Megatron-LM/commit/e4eb89b3bb3bb55c31606ee9d672038e17caa2a0) | 2026-10-21 | `refactor(training)`：将学习率与 WSD 步数解析迁移至 `SchedulerConfig.validate_run()` 与 `PretrainConfigContainer._resolve_scheduler_step_fields()`（#8211） |
| [`pytorch/torchtitan@48bbce4434`](https://github.com/pytorch/torchtitan/commit/48bbce4434dfac2c38fedc9c7ef9c1998b04cf50) | 2026-10-19 | `[llama3]`：为 Llama-3.2 1B/3B 显式配置 `scaling_factor=32.0`，消除 `17/32` 与 `35/64` 旋转频率失配及 `0.781 rad` 角度误差（#5528） |
| [`pytorch/torchtitan@efdde5edaf`](https://github.com/pytorch/torchtitan/commit/efdde5edaf2acad28782f5e8a851f9e9ecdb51d6) | 2026-10-19 | `[models]`：Attention Gym 为每个缺失模块生成独立 `_MissingKernel[<module>]` 子类与 `require_attn_gym(*kernels)` 精确报错（#5500） |
| [`LMCache/LMCache@18651e8f19`](https://github.com/LMCache/LMCache/commit/18651e8f19c94f56c32bc88e8826e37760a726c3) | 2026-10-20 | `Refactor legacy transfer channel under inprocess`：将 `CreateTransferChannel`、`NixlChannel` 与 `PySocketChannel` 统一收拢至 `lmcache.inprocess.v1.transfer_channel`（#5950） |
| [`ai-dynamo/dynamo@8e26089d3f`](https://github.com/ai-dynamo/dynamo/commit/8e26089d3f2ebddcad6c2117911c75d47a931ddf) | 2026-10-21 | `fix(sidecar)`：在 SGLang 与 TensorRT-LLM Sidecar 中转发可见停止词 ID（`stop_token_ids_visible`），保留 `<|call|>` 等工具调用定界符（#16759） |
| [`ai-dynamo/nixl@fda3a883c0`](https://github.com/ai-dynamo/nixl/commit/fda3a883c0aef00e3b42c4a0f08899aa0b49d3c6) | 2026-10-19 | `nixl_ep`：支持在 8 通道 / 8 线程 CPU Proxy（`FENCE_MODE=ep_based`）上运行专家并行，在 H100/GB200 上达到 Direct GPU RDMA 的 `100%–109%` 带宽（#2328） |
| [`kvcache-ai/Mooncake@a4fe83974b`](https://github.com/kvcache-ai/Mooncake/commit/a4fe83974b26e4565ee565d12db91f8af6e768c2) | 2026-10-19 | `[Conductor]`：新增 `conductor_c.h` C ABI 与 `mooncake-conductor/rust` 线程安全（`Send + Sync`）Rust 客户端绑定（#4651） |

---

## 2. 核心架构机制深挖

### 2.1 TransformerEngine GroupedMLP 自动对齐融合与 ONNX 0-D 假张量元数据（`9d9bd9e5c4`、`33f3b35619`）

1. **废弃环境变量门控，按 `split_alignment` 自动融合（`9d9bd9e5c4`）**：此前要在 Blackwell 上启用 `GroupedMLP_CuTeGEMMGLU` 融合，必须在导入 TE 前设置 `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`。PR #3584 将该环境变量标记为 Deprecated（若设置则自动将 `split_alignment` 覆写为 `256`），改为在 `OperationFuser` 中直接检查 `GroupedLinear` 声明的 `split_alignment`：对于 BF16/FP16 需满足 `split_alignment=256`，而对于 `MXFP8`、`NVFP4` 和 `FP8 block scaling` 仅需 `split_alignment=128`（配合 `ScaledSwiGLU(glu_interleave_size=32)`），并在仅因对齐不足错过融合时发出显式告警。
2. **ONNX FP8 Current-Scaling 假张量修正（`33f3b35619`）**：将 `onnx_cs_quantize_fp8_op.register_fake` 返回的缩放因子从 1-D `torch.ones(1, ...)` 修正为 0-D 标量 `torch.ones((), ...)`，使 `torch.library.opcheck(..., test_utils=("test_faketensor",))` 完全通过。

### 2.2 Megatron `SchedulerConfig` 迁移与 TorchTitan Llama-3.2 32x RoPE 修正（`e4eb89b3bb`、`48bbce4434`、`efdde5edaf`）

1. **Megatron `SchedulerConfig` 容器化验证（`e4eb89b3bb`）**：将 `arguments.py` 与 `training.py` 中散乱的基于迭代（`train_iters`）与基于样本（`train_samples` + 动态 `step_batch_size_schedule`）的 Warmup/Decay/WSD 步数推导收拢至 `PretrainConfigContainer._resolve_scheduler_step_fields()` 与 `SchedulerConfig.validate_run()`，并在 Checkpoint 加载时通过 `cfg.scheduler.override_opt_param_scheduler` 统一控制学习率上下界覆写。
2. **TorchTitan Llama-3.2 1B/3B `scaling_factor=32.0`（`48bbce4434`）**：Llama-3.2 1B 与 3B 官方 `config.json` 采用 `factor=32.0`（`low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8192`），而 TorchTitan 此前遗漏了该字段，回退到了 Llama-3.1 的默认值 `8.0`。这导致 1B 模型 **`17/32`** 个频率、3B 模型 **`35/64`** 个频率（所有波长高于 `2048` 的中低频分量）出现最高 `4x` 偏差，在位置 `2048` / `8192` 处产生 **`0.195 rad` / `0.781 rad`** 的最大旋转角误差，与 HuggingFace Transformers 对比的 logits 最大绝对误差达 `1.2e-4`（1B）/ `3.2e-4`（3B）。显式设置 `scaling_factor=32.0` 后，频率失配归零，logits 最大绝对误差降至 **`7e-7`**（浮点舍入极限）。
3. **Attention Gym 按模块 `_MissingKernel` 子类化（`efdde5edaf`）**：当 `attn_gym.sparse` 因缺少 `cutlass` 导入失败而 `GDN` 所需的 `triton` 正常时，旧版门面会统一抛出第一个记录的 `cutlass` 错误。PR #5500 为每个模块动态派生 `_MissingKernel[<module>]` 子类并配合 `is_missing_kernel()` `TypeGuard`（防止混入 SPMD 局部 Autograd 注册表），使 `KDAKernel`、`GDNKernel`、`ContextParallelInnerGDN` 等仅在自身依赖的特定后端缺失时精准报错。

### 2.3 NIXL EP 8x8 CPU Proxy、Dynamo 可见停止词转发与 Mooncake Conductor Rust 绑定（`fda3a883c0`、`8e26089d3f`、`18651e8f19`、`a4fe83974b`）

1. **NIXL EP 8 通道 / 8 线程 CPU Proxy（`fda3a883c0`）**：并非所有网卡/驱动或 RoCE 虚拟函数都支持 UCX Direct GPU-Initiated RDMA。PR #2328 支持通过 `NIXL_EP_DEVICE_MODE=proxy` 在不改动任何 GPU 内核的前提下，由 GPU 向 `proxy_channel_count=8` 个环形命令队列写入请求，并由绑定在同 NUMA 节点的 `proxy_thread_count=8` 个自旋 CPU 代理线程配合 `engine_config=FENCE_MODE=ep_based` 推进 UCX：
   - 在 `128 tokens/GPU, H=7168, top-8, E=256` 纯网络测试中，`8 channels × 8 threads` 在 **H100 2/4/8 卡上达到 Direct 带宽的 `103%`（`59.0 GB/s`）、`104%`（`45.5 GB/s`）、`100%`（`41.6 GB/s`）**；在 **GB200 2/4 卡上达到 `109%`（`53.3 GB/s`）、`99%`（`36.5 GB/s`）**；在 **GB300 2/4 卡上达到 `104%`（`50.7 GB/s`）、`88%`（`36.8 GB/s`）**。
   - 同时揭示了关键容量边界：在 H100 4 卡共享 32 核 NUMA 节点上，若将自旋线程盲目翻倍至 `16 threads/GPU`（共 64 线程 > 32 核），核心超卖会导致吞吐从 `45.5 GB/s` 暴跌至 **`1.2 GB/s`（仅 `3%`）**！
2. **Dynamo Sidecar 可见停止词（`8e26089d3f`）与 LMCache `inprocess` 通道（`18651e8f19`）**：Dynamo SGLang Sidecar 将 `stop_token_ids_visible` 合并进 `stop_token_ids` 并置 `return_text = None`（因为 SGLang 仅裁剪文本而保留 `output_ids` 中的停止词），TRT-LLM Sidecar 则开启 `include_stop_in_output = Some(true)` 并在 `ResponseState` 中暂存非终止隐式停止词，确保 Harmony `<|call|>` 等工具调用定界符完整送达前端解码器；LMCache 将 `NixlChannel`、`MockMemoryChannel`、`PySocketChannel` 统一收拢至 `lmcache.inprocess.v1.transfer_channel`。
3. **Mooncake Conductor C ABI 与 Rust 客户端（`a4fe83974b`）**：通过 `conductor_c.h` 暴露 `create/destroy/setup/close/health_check/query/register/unregister/get_global_view/list_services` 10 组 C ABI 与 7 级错误码（`0` 至 `-2001`），并在 `mooncake-conductor/rust` 中基于 `bindgen` 静态链接 `libconductor_cpp_core.a`，提供 `Send + Sync` 的强类型 `ConductorClient`。
