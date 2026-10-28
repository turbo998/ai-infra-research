# Phase 126 Track C: Megatron-LM 单 DP 组 Refit 均衡与全阶段 Sequence Packing、TorchTitan 泛型 Loss 抽象、veRL VeOmni FP8 Rollout 量化器、Mooncake `hccs:device` 2 MiB 门控、LMCache `ObjectKeyIdentity` 与 Dynamo 流释放

> **Phase 126 (`priority: 135`) · 证据等级**: 上游源码与提交说明 (`implementation`) + 本项目离线合成标定 (`own-offline`)。未声明物理多节点集群实测 (`own-hardware`)。

---

## 1. 核心架构问题与源码级解构

### 1.1 Megatron-LM `get_batch` 配置缓存、全 PP 阶段 Sequence Packing 与单 DP 组 Refit 轮询负载均衡 (`#8525`, `#8592`, `#8502`)
- **`get_batch` 中缓存 `TransformerConfig` (`#8525`, `51039e52`)**:
  - **上游路径**: `pretrain_gpt.py`、`pretrain_hybrid.py` (`51039e52714e2f664ebfcf4c25789f6c94ea8fa1`)。
  - **机制**: `get_batch` 在每个 microbatch 都会执行；此前每次调用都重建 `TransformerConfig` 并重复触发 `__post_init__` 校验与日志告警。引入 `_get_transformer_config(args)` 按 `args` 对象身份缓存后，单 step（64 microbatches）配置构建次数从 `64` 降至 **`1`**。
- **全 Pipeline 阶段构建 Sequence-Packing 数据集 (`#8592`, `e4a643d8`)**:
  - 修复 `is_dataset_built_on_rank`：只要 `is_packed_sequence` 或 `args.sequence_packing_scheduler is not None` 生效，所有 PP 阶段（当 `tp_rank == 0` 时）均构建 packed-sequence 数据集（中间 PP stage 缺失数据集数降至 `0`）。
- **单 DP 组下的 RL Weight Refit 源副本轮询均衡 (`#8502`, `4a55e6fe`)**:
  - **问题根因**: `megatron/core/resharding/utils.py` 的 `_round_robin_dp` 原先包含一条错误的快速路径：`if len(grouped_by_dp) == 1: return src_meta_list[0]`。当训练侧配置为 `TP=1, PP=1, DP=8` 时，所有 8 个源副本同属唯一个 DP group（`len(grouped_by_dp) == 1`），该快速路径导致所有目标推理副本全部从 `src_meta_list[0]`（单源 rank 0）拉取权重，引发严重的单网卡发送瓶颈。
  - **修复机制**: 移除该短路返回，对单 DP 组内的多源副本同样按目标 rank 执行轮询分摊。在 `DP=8`、`64 GB` 权重同步、`50 GB/s` (400 Gbps) 单 rank 带宽下，离线标定 refit 传输时间从 `10.24 s` 降至 **`1.28 s`（`8.0x` 加速）**。

### 1.2 TorchTitan 泛型 `Loss[LossInputT]`、Qwen3 Typed-Decision 微调与 veRL VeOmni 训练侧 Rollout 权重量化器 (`#4582`, `#5569`, `verl#8369`)
- **TorchTitan `Loss(ABC, Configurable, Generic[LossInputT])` 抽象重构 (`#4582`, `740d5f75`) & Qwen3 Typed-Decision Recipe (`#5569`, `36af428d`)**:
  - **问题根因**: 旧版 `BaseLoss` / `CrossEntropyLoss` 假定单一张量输入，导致操作预测/标签元组的 `MTPLoss`（多 token 预测组合损失）与改变执行方式的 `ChunkedLossWrapper` 继承层次失真。
  - **设计契约**: 引入逆变泛型参数 `LossInputT = TypeVar("LossInputT", contravariant=True)`，统一 `__call__(self, pred: LossInputT, labels: LossInputT, global_loss_token_counts: torch.Tensor | None = ...)` 签名，并在 `DAPOLoss` 等 RL 损失中保持与 vLLM 温度缩放 logprobs 的逐位对齐；同时新增 `torchtitan_recipes.decision` 支持 Qwen3 Jev 风格类型化决策微调。
- **veRL `VeOmniEngineWithLMHead` 支持训练侧 Rollout 权重量化器 (`verl#8369`, `6f3f449e`)**:
  - **问题根因**: 自 veRL 统一 QAT 与低精度 rollout (`#8287`) 后，当 checkpoint 的 `quantization_config` 为 FP8（如 `DeepSeek-V4-Flash-Base`）时，`init_model` 会派发 `Fp8ConfigWeightQuantizer` 并调用 `engine.set_rollout_weight_quantizer`，但此前仅 Megatron 引擎实现了该接口，导致所有 VeOmni 引擎任务（如 `run_deepseek_v4_veomni.sh`）在启动时直接抛出 `NotImplementedError`。
  - **修复机制**: 在 `verl/workers/engine/veomni/transformer_impl.py` 中保存 `self._rollout_weight_quantizer` 并将其包裹于 `get_per_tensor_param` 参数流上：对已导出混合 FP8/FP4 布局的 `deepseek_v4` / `deepseek_v41` 执行零拷贝 passthrough，对其他 BF16 模型流执行在线 FP8 量化（初始化失败数 `1 -> 0`，BF16->FP8 rollout 权重同步流量削减 **`50.0%`**）。

### 1.3 Mooncake `hccs:device` 2 MiB 对齐门控、Ascend 分片失败传播、TENT 自观测契约、LMCache `ObjectKeyIdentity` 与 Dynamo 流释放 (`#5049`, `#4888`, `#4938`, `lmcache#5988`, `lmcache#6090`, `dynamo#17511`, `dynamo#14325`)
- **Mooncake `hccs:device` 2 MiB 对齐门控 (`#5049`, `811c97f0`) & Ascend 设备解析失败分片报错 (`#4888`, `eccc7c59`)**:
  - 修复 `#4903` 将所有非 RoCE 注册一律视为 HCCS 的缺陷：新增 `HasHccsDeviceProtocolDescInGlobalResourceConfig` 并在安装时捕获 `hccs_device_protocol_`，仅当角色解析后的 `protocol_desc` 明确包含 `"hccs:device"`、关闭 fabric mem 且缓冲区可远端访问时才强制执行 `2 MiB` 对齐检查（`"hccs:host"` 与 `"roce:*"` 误拒绝数降为 **`0`**）。
  - 在 `AscendDirectTransport::submitTransferTask` 中，当 `ResolveCurrentEngineId` / 设备查找拒绝提交时，显式将所有关联 `Slice*` 分片标记为失败状态，防止上层调用无限挂起等待。
- **Mooncake TENT 指标自观测契约 (`#4938`, `d3ef7b1c`)**: 当 Prometheus 或 JSON 序列化失败时返回 HTTP 500，并在下一次成功抓取中暴露失败计数与 HTTP 状态。
- **LMCache `ObjectKeyIdentity` 不可变身份抽取 (`lmcache#5988`, `0108f5b6`) & Dynamo OTLP Trace 解析 (`lmcache#6090`, `b89bc5f9`)**:
  - 将 `chunk_hash`、`kv_rank`、`model_name`、`object_group` 与 `cache_salt` 抽取为冻结的 `ObjectKeyIdentity`，在本地 `ObjectKey` 与内部 MP 线协议消息间共享同一身份对象，消除冗余字段拷贝（节省 `50.0%` 键元数据副本），并在 `lmcache bench trace-analyze` 中原生支持 `dynamo.request_trace` OTLP 日志格式。
- **Dynamo 转发 OpenAI `logit_bias` (`dynamo#17511`, `843dc327`) & 本地停止后立即释放 vLLM 流 (`dynamo#14325`, `78ac1bc9`)**:
  - 在 `components/src/dynamo/sglang/request_handlers/llm/decode_handler.py` 中将 `"logit_bias"` 加入采样透传白名单，修复此前被静默丢弃的问题；并在 `vllm_processor.py` 中追踪 `locally_finished`，一旦本地提前触发 stop 条件立即释放后端 vLLM 生成流（泄漏流数降至 **`0`**）。

---

## 2. 离线合成实验标定 (`own-offline`)

实验脚本：`experiments/megatron_get_batch_seqpack_refit_dp_torchtitan_loss_verl_veomni_mooncake_lmcache_dynamo_phase126/run.py`

| 子系统 / 指标 | 修复前 (Before) | Phase 126 修复后 (After) | 关键架构收益 |
|---|---|---|---|
| Megatron-LM 单 DP 组 (`DP=8`) Weight Refit | 全部从 `src_meta_list[0]` 拉取 (`10.24 s`) | **单 DP 组内多源轮询均衡 (`1.28 s`)** | **`8.0x`** Refit 吞吐加速 |
| Megatron-LM `get_batch` 配置构建 & PP Packing | 每 step 重建 `64` 次 / 中间 PP 缺数据集 | **`_get_transformer_config` 缓存 (`1` 次)** | 消除日志风暴与中间 PP 阶段断流 |
| veRL VeOmni DeepSeek-V4 / FP8 Rollout | `NotImplementedError` 启动崩溃 | **`set_rollout_weight_quantizer` + Passthrough** | **`0`** 启动失败；BF16->FP8 同步流量 **`-50.0%`** |
| Mooncake Ascend `2 MiB` 对齐与分片提交 | `"hccs:host"` 误判 2 MiB / 设备失败静默挂起 | **仅 `"hccs:device"` 门控 + 分片显式 Fail** | **`0`** 误拒绝 / **`0`** 挂起分片 |
| LMCache `ObjectKeyIdentity` & Dynamo Stop | 重复拷贝 5 字段 / 本地 stop 后流驻留 | **共享不可变 `ObjectKeyIdentity` + 立即释放流** | 键字段副本 **`-50.0%`**；泄漏 vLLM 流 **`0`** |
