# TorchTitan SimpleFSDP 参数检查守卫、Kimi-K3 MXFP4 权重适配器与 Megatron-LM 空闲重跑状态严格加载修复

> **English Version**: [torchtitan-fsdp-inspection-guard-kimi-k3-mxfp4-adapter-megatron-idle-rerun.en.md](torchtitan-fsdp-inspection-guard-kimi-k3-mxfp4-adapter-megatron-idle-rerun.en.md)  
> **证据等级**: `implementation`（源码审计）+ `own-offline`（确定性离线建模）  
> **对应实验**: [`experiments/torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun/run.py`](../../experiments/torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun/run.py)

---

## 1. 工程背景与核心故障

在大规模分布式训练（FSDP + Expert Parallelism + Pipeline Parallelism）与量化权重检查点转换中，张量参数化生命周期与严格状态字典（Strict State-Dict）契约极易产生隐蔽冲突：
1. **TorchTitan GraphTrainer `meta` 设备 FSDP 检查误触发 MXFP8 实例化（`pytorch/torchtitan@3b1714ab`，`#5560`）**：在 B200 `graph_trainer_dist_moe_mxfp8_fsdp_ep_pp` 任务中，当对已被 GraphPP 参数化的共享模块再次调用 `simple_fsdp.data_parallel()` 时，`linear_param_shard_placements(model)` 读取 `child.weight` 会触发 `ReplicateComputation` 并调用 `_build_operands`。此时模型仍处于 `meta` 设备（尚未执行 `to_empty(device=...)`），导致 `dist_moe` 因 `weight.is_cuda == False` 抛出 `ValueError: weight must be contiguous CUDA storage`。
2. **TorchTitan Kimi-K3 `compressed-tensors` MXFP4 检查点双向映射与 KDA 布局保持（`pytorch/torchtitan@7d563223`，`#5337`）**：开源 Kimi-K3 检查点跨 `96` 个 safetensors 文件存储 `249,924` 个张量名，将路由专家存为 `weight_packed` / `weight_scale`，并将 KDA 注意力的 `A_log` 从 `[n_heads]` 填充至 `[head_dim]`、将 `dt_bias` 展平为一维。
3. **Megatron-LM `dist_ckpt_strictness="raise_all"` 下空闲 `rerun_state_machine` 加载失败（`NVIDIA/Megatron-LM@e564873e`，`#7222`）**：为复用缓存的保存计划（Cached Save Plans），空闲状态（`RerunState.NOT_RUNNING_YET`）的检查点依然写入了 `rerun_state_machine` 分片；但加载端 `validate_state_dict` 对空闲状态返回 `False`，导致严格加载时未将该键纳入预期 schema 而报未预期键错误。

---

## 2. 一手源码机制剖析

### 2.1 TorchTitan `disable_active_parametrization()` 检查守卫
在 `torchtitan/experiments/graph_trainer/simple_fsdp.py` 的 `data_parallel()` 中：
```python
if param_shard_placements is None:
    with disable_active_parametrization():
        param_shard_placements = linear_param_shard_placements(model)
```
通过在扫描线性层分片布局时包裹 `disable_active_parametrization()`，检查逻辑直接读取底层的 `_ShardedFSDPTensor` 原始参数而非执行前向计算参数化（`ReplicateComputation`），将 `meta` 设备上的 `_build_operands` 调用次数从 `1` 降至 `0`。

### 2.2 Kimi-K3 MXFP4 `QuantizedHuggingFaceStorageReader` 与 KDA DTensor 重分布
在 `torchtitan/models/kimi_k3/state_dict_adapter.py` 中：
- **量化读取器与 96 文件索引重映射**：`get_hf_storage_reader(..., from_quantized=True)` 返回 `QuantizedHuggingFaceStorageReader(path=path, thread_count=4, target_dtype=torch.bfloat16)` 在读取时将 MXFP4 专家反量化为 BF16；初始化时将 `fqn_to_index_mapping` 中的 `X.weight_packed` 映射为 `X.weight` 并剔除 `X.weight_scale`（保留原始 `96` 文件布局、`249,924` 个 HF 键名），同时在纯文本配置（`vision_encoder is None`）下过滤 `vision_tower.*` 与 `mm_projector.*`，避免 DCP 合并写出空 dtype 头。
- **KDA `A_log` 与 `dt_bias` DTensor 坐标保护**：在 `to_hf()` 中，由于填充（`A_log` 在 `to_quantized_layout=True` 下填充 `head_dim - n_heads = 128 - 16 = 112` 个元素）和展平（`dt_bias` 从 `[16, 128]` 展平为 `[2048]`）会改变张量坐标，适配器先保存 `(value.device_mesh, value.placements)` 并将 DTensor 临时重分布为 `(Replicate(),) * ndim`；在 `from_hf()` 切片 `value[:n_heads]` / 重塑 `[n_heads, head_dim]` 后，再调用 `value.redistribute(device_mesh=layout[0], placements=layout[1])` 精确恢复原始分布式放置。

### 2.3 Megatron-LM 空闲重跑状态严格加载与恢复解耦
在 `megatron/training/checkpointing.py` 中：
```python
restore_rerun_state = rerun_state_machine.validate_state_dict(saved_rerun_state)
idle_rerun_state = (
    isinstance(saved_rerun_state, dict)
    and saved_rerun_state.get("state") == RerunState.NOT_RUNNING_YET
)
if restore_rerun_state or idle_rerun_state:
    gen_sd_rerun_state = rerun_state_machine.state_dict(
        data_iterator=None, ckpt_format=ckpt_format, force=True
    )
    ignore_rerun_state = not restore_rerun_state
```
将“在分布式检查点 schema 中声明 `rerun_state_machine` 键（`restore_rerun_state or idle_rerun_state`）”与“恢复重跑状态机（`ignore_rerun_state = not restore_rerun_state`）”解耦：当检查点处于 `NOT_RUNNING_YET` 时，`strict=True`（`raise_all`）正常吸收该分片，同时 `ignore_rerun_state=True` 确保加载后不会误触发重跑（`rerun_requested == False`）。

---

## 3. 架构师设计练习与可证伪问题

1. **设计分析题**：为何在 `KimiK3StateDictAdapter.to_hf()` 中，只有当 `to_quantized_layout=True` 时才对 `A_log` 填充 `head_dim - n_heads`？
   - **解答**：因为开源的 MXFP4 权重文件将 `A_log` 填充到了 `head_dim`（128），DCP 在加载规划（Load Planning）阶段要求目标张量形状与检查点一致；而在导出标准未量化 HF 检查点（`to_quantized_layout=False`）时，HuggingFace 模型定义期望 `A_log` 为未填充的 `[n_heads]`（16）。
2. **可证伪假设**：如果在 `dist_moe` 边界仅对权重调用 `.contiguous()` 而不禁用 `SimpleFSDP` 活动参数化，能否修复 B200 GraphPP + FSDP 的构建报错？（不能，上游 CI 实测证明报错原因是 `weight.is_cuda == False` 即张量仍在 `meta` 设备上，而非内存步长非连续）。
