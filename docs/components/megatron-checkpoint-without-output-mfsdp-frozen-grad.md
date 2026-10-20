# Megatron-LM 激活与参数显存生命周期：`CheckpointWithoutOutput` 四态自动机零环存储共享与 `Megatron-FSDP` 冻结模块延迟清理 / 微批次重叠

> **English Version**: [megatron-checkpoint-without-output-mfsdp-frozen-grad.en.md](megatron-checkpoint-without-output-mfsdp-frozen-grad.en.md) | **返回索引 / Back to README**: [README.md](../../README.md) | [README.en.md](../../README.en.md)

## 1. 工程问题定义：大模型训练中两类隐蔽的显存生命周期与流同步缺陷

在 `NVIDIA/Megatron-LM` 的超大规模 MoE / 多模态训练中，激活重计算（`CheckpointWithoutOutput`）与全分片数据并行（`Megatron-FSDP`）存在两个深层生命周期问题（对应提交 `d2c430ab485c6947701447e5d551d3aafaa39161` 与 `d37b3a8fdb37af1b3a1d75720636af4c323bd522`）：

1. **`CheckpointWithoutOutput` 的自动微分引用环与 Offload 解包冗余**：
   - `CheckpointWithoutOutput` 广泛用于 MoE Router、Shared Expert 与 LayerNorm：前向在 `torch.no_grad()` 下计算输出并交给下游算子消费，随后调用 `discard_output_and_register_recompute` 将输出张量的底层存储缩容为 `0`（`untyped_storage().resize_(0)`），待反向到达 `hook_tensor` 时重跑函数并原地恢复底层存储指针。
   - 旧版实现将 `ctx` 与 `outputs` 相互持有在 `CheckpointWithoutOutput` 门面对象上：一旦某个重计算区域因跳过反向（如评估、早停或无梯度分支）未触发 `_recompute`，`output -> grad_fn -> ctx -> checkpoint_obj -> outputs` 将形成**跨 Autograd 图的强引用环**，导致整块计算图与激活无法被 GC 回收！同时，需要保证细粒度激活卸载（`saved_tensors_hooks`）只对输入解包一次，且下游算子保存的视图（如 `TE GroupedLinear` 的 `inp.reshape()` + `torch.split()`）在重建后无需修改 `TensorImpl` 版本计数器即可零拷贝看到新数据。
2. **`Megatron-FSDP` 全冻结模块的提前释放（Use-After-Free）与微批次 `ReduceScatter` 过度等待**：
   - 当一个 FSDP 单元仅包含**冻结权重**（`_trainable_parameter_countdown.initial_value == 0`，如多模态冻结视觉塔或 RL 冻结参考策略层），且其输入不需要梯度（`all(grad is None for grad in grad_input)`，但内部有独立分片的可训练子模块保持反向图存活）时，PyTorch 的 `register_full_backward_hook` 会在**内部反向计算尚未执行前**提前触发！旧版立即调用 `post_backward_hook` 将未分片冻结权重缩容为 `0`，导致内部算子反向读取已释放显存（CUDA Use-After-Free）！
   - 此外，在梯度累加的非末微批次（`microbatch(context, is_last=False)`）结束时，旧版 `context.post_backward()` 无条件调用 `finish_grad_sync()`（`current_stream().wait_stream(reduce_scatter_stream)`），强制阻塞计算流等待 `ReduceScatter` 流，扼杀了微批次 $m$ 的尾部 `ReduceScatter` 与微批次 $m+1$ 前向计算之间的跨微批次重叠窗口。

---

## 2. 一手源码走读：`_RecomputeRegion` 四态自动机与 `Megatron-FSDP` 延迟清理

### 2.1 `_RecomputeRegion` 四态前向自动机与 `UntypedStorage` 零拷贝共享（`megatron/core/tensor_parallel/random.py`）

提交 `d2c430ab485c6947701447e5d551d3aafaa39161`（`megatron/core/tensor_parallel/random.py` L773–1195）将状态从 `CheckpointWithoutOutput` 门面彻底剥离到单次调用专属的 `_RecomputeRegion`，定义单向四态枚举 `_RegionState`：

$$\texttt{RECORDED} \xrightarrow{\text{release()}} \texttt{RELEASED} \xrightarrow{\text{rebuild()}} \texttt{REBUILT} \xrightarrow{\text{backpropagate()}} \texttt{DONE}$$

| 阶段 / 组件 | 核心机制（`random.py` L838–1175） | 解决的架构痛点 |
|---|---|---|
| **`_InputStash` (`L838–859`)** | 通过独立自定义 `autograd.Function` 调用 `ctx.save_for_backward(*tensors)` 返回空 Token `torch.empty(0)`；在 `rebuild()` 时仅通过 `stash.grad_fn.saved_tensors` 读取一次并置 `self._stash = None` | 使激活卸载 Hook (`saved_tensors_hooks`) 与 CUDA Graph 观察器正常感知输入，且**严格只触发一次 Unpack（杜绝 CPU Offload 双重 H2D 重载）** |
| **`release()` 与 `_retained_storage_ptrs` (`L926–941`)** | 保存 `output.detach()` 别名（断开 `grad_fn` 引用环），对不在 `_retained_storage_ptrs` 中的输出执行 `storage.resize_(0)`；门面对象将 `self._region` 置 `None` 并仅保留 **`weakref.ref(region)`** | 保护与输入共享存储的输出（如 TE `MakeExtraOutput`），且当反向未执行时区域随计算图立即被 GC 回收（**零引用环泄漏**） |
| **`_share_rebuilt_storages()` (`L979–1000`)** | 在 `UntypedStorage` 层（位于 `TensorImpl` 之下）使已释放输出的 `StorageImpl` 指向重跑输出的数据指针（当两输出共享同一缓冲时仅重指向一次） | 下游保存的**所有 `view` / `reshape` / `split` 自动看到新数据**，且**不递增 Tensor `version_counter`**（Autograd 零报错） |

### 2.2 `Megatron-FSDP` 冻结模块延迟清理与非末微批次异步重叠（`d37b3a8fdb37`）

在提交 `d37b3a8fdb37af1b3a1d75720636af4c323bd522` 中：
1. **冻结权重无输入梯度时的延迟清理栈（`fully_shard.py` L200–213 & `context.py` L99–104, L226–231）**：
   - 当 `self._trainable_parameter_countdown.initial_value == 0` 时，注册 `grad_checking_post_backward_hook`：
     ```python
     def grad_checking_post_backward_hook(module, grad_input, _grad_output):
         if all(grad is None for grad in grad_input):
             module.context._delayed_post_backward_callbacks.append((module, post_backward_hook))
         else:
             post_backward_hook(module)
     ```
   - 待整个反向图的根回调触发 `context.post_backward()` 时，调用 `_run_delayed_post_backward_callbacks()` 按 **LIFO（`pop()` 先子模块后父模块，正确闭合嵌套 NVTX Range）** 执行延迟清理，确保内部反向计算期间冻结权重 `100%` 保持有效（`32 MiB` 未释放），反向完成后才缩容为 `0 B`。
2. **非末微批次跳过 `finish_grad_sync()`（`context.py` L215–224 & `fully_shard.py` L177–190）**：
   - `context.post_backward()` 将流同步条件收紧为：
     ```python
     if not self.caller_managed_grad_sync and self.is_last_microbatch:
         self.finish_grad_sync()
     ```
   - 在 `with microbatch(context, is_last=False):` 作用域内，每个微批次结束后仍立即释放未分片权重（保障峰值显存不涨），但**跳过 `current_stream().wait_stream(reduce_scatter_stream)`**；对于 $M=4$ 个梯度累加微批次，流同步阻塞从 `4` 次降至 **`1` 次**，释放出 **`3` 个跨微批次 `ReduceScatter -> Next Forward` 重叠窗口**！

---

## 3. 确定性实验验证（`experiments/megatron_ckpt_without_output_mfsdp_sim/run.py`）

运行 `python experiments/megatron_ckpt_without_output_mfsdp_sim/run.py --check` 验证：
- `CheckpointWithoutOutput`：记录后主输出 `64 MiB`，`release()` 后主输出降为 `0 B` 而输入别名输出保持 `16 MiB`；两次 Hook 触发下 `unpack_count == 1`，重建后下游视图读取到 `0x3000` (`64 MiB`) 且 `version_counter == 0`；抛弃区域的 `weakref` 在 GC 后返回 `None`。
- `Megatron-FSDP`：修复前内部反向期间冻结权重为 `0 B`（触发 Use-After-Free），修复后保持 `32 MiB` 并在 `post_backward()` 按 `["child_frozen_module", "parent_frozen_module"]` 顺序归零；4 微批次下流等待从 `4` 次降至 `1` 次。

---

## 4. 一手来源与证据链索引

| Source ID | 仓库与提交 / 文件路径 | 证据等级 |
|---|---|---|
| `megatron-ckpt-without-output-region-2026` | `NVIDIA/Megatron-LM` commit `d2c430ab485c6947701447e5d551d3aafaa39161` (`megatron/core/tensor_parallel/random.py`, blob `578e2f5244d4`) | `implementation` |
| `megatron-fsdp-frozen-cleanup-microbatch-2026` | `NVIDIA/Megatron-LM` commit `d37b3a8fdb37af1b3a1d75720636af4c323bd522` (`megatron_fsdp/experimental/context.py`, blob `3013d39e3a12`; `fully_shard.py`, blob `527efff8585c`) | `implementation` |
| `megatron-fsdp-schedule-plan-cleanup-2026` | `NVIDIA/Megatron-LM` commit `d37b3a8fdb37af1b3a1d75720636af4c323bd522` (`megatron/core/models/common/model_chunk_schedule_plan.py`, blob `9e78dc145827`) | `implementation` |
