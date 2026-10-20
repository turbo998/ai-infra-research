# Megatron-LM `_RecomputeRegion` Zero-Cycle Storage Sharing & `Megatron-FSDP` Frozen Cleanup Simulator / 模拟器

## 中文说明

本确定性标准库模拟器（`run.py`）基于 `NVIDIA/Megatron-LM` 提交 `d2c430ab485c` 与 `d37b3a8fdb37` 验证：
1. **`CheckpointWithoutOutput` 四态前向自动机 (`_RegionState: RECORDED -> RELEASED -> REBUILT -> DONE`)**：验证 `_InputStash` 仅解包一次输入（避免 CPU Offload 双重重载）、保留输入别名存储（`_retained_storage_ptrs`，如 TE `MakeExtraOutput`）、门面弱引用 `weakref.ref(region)` 消除自动微分引用环，以及在 `UntypedStorage` 层零拷贝重指向重建数据（下游视图自动可见且不递增 Tensor `version_counter`）。
2. **`Megatron-FSDP` 冻结模块延迟清理与非末微批次 `ReduceScatter` 重叠**：当全冻结 FSDP 单元（`_trainable_parameter_countdown == 0`）无输入梯度（`all(g is None for g in grad_input)`）时，`grad_checking_post_backward_hook` 将清理回调推入 `_delayed_post_backward_callbacks` 并按 LIFO（先子后父）在 `post_backward()` 执行，根除内部反向读取已释放显存（Use-After-Free）；同时在 `microbatch(is_last=False)` 下跳过 `finish_grad_sync()` 流等待，跨微批次释放 `N-1` 次 `ReduceScatter` 与前向计算的并发重叠窗口。

## English Summary

This deterministic Python standard-library simulator (`run.py`) models `NVIDIA/Megatron-LM` commits `d2c430ab485c` and `d37b3a8fdb37`:
1. **`CheckpointWithoutOutput` 4-State Automaton (`_RegionState`)**: Verifies single-unpack `_InputStash`, `_retained_storage_ptrs` alias preservation, `weakref.ref(region)` cycle-free garbage collection on skipped backward passes, and `UntypedStorage`-level zero-copy pointer sharing without bumping autograd version counters.
2. **`Megatron-FSDP` Frozen-Module Deferred Cleanup & Microbatch `ReduceScatter` Overlap**: Models `grad_checking_post_backward_hook` LIFO deferral (`_delayed_post_backward_callbacks`) preventing use-after-free when `grad_input` is all `None`, and quantifies the `N-1` stream-wait reductions skipped across non-final microbatches (`is_last=False`).

```bash
python experiments/megatron_ckpt_without_output_mfsdp_sim/run.py --check
```
