# SGLang PD+DP 两阶段角色切换、TRT-LLM Attention-DP KV 事件通信对齐与 Mooncake 零拷贝 VMM 边界切分深挖

> **English Companion**: [sglang-pd-dp-attention-role-switch-trtllm-dp-kv-event-flush-mooncake-zerocopy.en.md](sglang-pd-dp-attention-role-switch-trtllm-dp-kv-event-flush-mooncake-zerocopy.en.md)  
> **Phase 41 (`priority: 50`) · 确定性复现实验**: [`experiments/sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc/run.py`](../../experiments/sglang_pd_dp_role_switch_trtllm_adp_kv_mooncake_zc/run.py)  
> **7-Card 4:5 视觉报告**: [`docs/reports/ai-infra-phase41-benchmark-cards.html`](../reports/ai-infra-phase41-benchmark-cards.html)

---

## 一、核心架构结论（Executive Summary）

1. **SGLang PD 分离 + DP / DP Attention 两阶段运行时角色切换（`PR #44215` / `af49ada16d`）**：
   - 解除 `pd_disaggregation_hook.py` 中对 `dp_size > 1` 和 `--enable-dp-attention` 的静态阻断（仅保留拒绝 `--enable-dp-attention-local-control-broadcast`），首次支持多 DP Rank 下的在线 Prefill ↔ Decode 角色热切换。
   - **两阶段 `prepare_only=True` 预检防脑裂协议**：当 `num_dp_ranks > 1` 时，`tokenizer_control_mixin.pd_role_switch` 先向所有 DP Worker 广播 `PdRoleSwitchReqInput(prepare_only=True)`，校验各 Rank 空闲状态及目标角色 CUDA Graph 显存余量（返回 `message="ready", safe_to_restore=True`）。若任一 Rank 显存不足（如 Rank 2 失败），在**任何 Worker 销毁 KV Cache / Transfer Backend / CUDA Graph 之前**无损中止切换（`ranks_torn_down = 0, split_brain = False`），彻底消除单阶段广播导致的 `3/4` Rank 撕裂脑裂。
2. **TensorRT-LLM Attention-DP 重叠调度器 KV Cache 事件集合通信对齐（`PR #20314` / `ec782566f5`）**：
   - 修复 `_executor_loop_overlap` 与 `_consume_previous_batch_for_rebalance` 在启用 `enable_attention_dp=True` 与 `enable_kv_cache_events=True` 时的多卡集合通信死锁。
   - 将 `self._add_kv_cache_events()`（内含跨 ADP Rank 的 `kv_event_allgather`）从仅 `processed` 分支执行的 `_process_previous_batch()` 移出至外层对称主干，并在 `_consume_previous_batch_for_rebalance` 中强制无本地 Batch（`absent`）的 ADP Rank 同步执行 `_flush_pending_transfer_responses()`、`_enqueue_responses([])` 与 `kv_event_allgather`，使 `("absent", "processed")`、`("processed", "deferred")`、`("deferred", "absent")` 全场景 100% 免死锁。
3. **TensorRT-LLM Mooncake Store 零拷贝 GPUDirect RDMA 与 `cuMemCreate` 物理边界切分（`PR #20108` / `69e06b8afb`）**：
   - 将 `MooncakeStoreConnectorConfig.stage_through_host` 默认值从 `True` 翻转为 **`False`**（默认启用 GPU 显存零拷贝直传）。
   - 针对 V2 KV Cache 管理器由多个 `cuMemCreate` 物理分配块拼接而成的虚拟地址池（GPUDirect RDMA 禁止单个 `ibv_reg_mr` / `ibv_reg_dmabuf_mr` 跨越两个 `cuMemCreate` 物理块），在 `PageAddressing` 中通过 `merge_intervals` 合并滑动窗口/异构层重叠区间，再用 `allocation_spans` 与 `split_at_cuts` 沿物理分配边界精确切分注册区间与单页传输片段；同时在无 `nvidia_peermem` 的 GB200/GB300 Open-Kernel 节点自动注入 `WITH_NVIDIA_PEERMEM=0` 启用 DMA-BUF，并在 8 卡节点上省去 **`1,024 MiB`**（`8 ranks * 2 directions * 64 MiB`）Pinned Host DRAM 暂存池。

---

## 二、一手上游提交与证据映射（First-Hand Upstream Ledger）

| 上游仓库 | 提交 / PR | 日期 | 核心改动模块 | 证据 ID |
|---|---|---|---|---|
| `sgl-project/sglang` | [`af49ada16d` (`#44215`)](https://github.com/sgl-project/sglang/commit/af49ada16d9b4e7535fd6ff542247f50b8c92c9f) | `2026-10-21` | `python/sglang/srt/disaggregation/role_switch.py`, `tokenizer_control_mixin.py` | `EV-2026-358` |
| `NVIDIA/TensorRT-LLM` | [`ec782566f5` (`#20314`)](https://github.com/NVIDIA/TensorRT-LLM/commit/ec782566f50ad9548e99818fa1d74c7c66cc572d) | `2026-10-21` | `tensorrt_llm/_torch/pyexecutor/py_executor.py` | `EV-2026-359` |
| `NVIDIA/TensorRT-LLM` | [`69e06b8afb` (`#20108`)](https://github.com/NVIDIA/TensorRT-LLM/commit/69e06b8afbffc4726a63b43fb4acb9fda340f3a7) | `2026-10-21` | `tensorrt_llm/_torch/pyexecutor/connectors/mooncake_store/{addressing.py,gpudirect.py}` | `EV-2026-360` |

---

## 三、架构机制深挖（Architectural Deep Dive）

### 3.1 SGLang PD+DP 两阶段角色切换与 TRT-LLM ADP 集合通信对称化

```
[SGLang TokenizerControlMixin.pd_role_switch (num_dp_ranks = 4)]
  Phase 1: Broadcast PdRoleSwitchReqInput(prepare_only=True)
    ├─► Rank 0,1,3: idle & CUDA graph headroom OK -> (success=True, message="ready")
    └─► Rank 2: OOM headroom check fails        -> (success=False, safe_to_restore=True)
  Gate: not all(success) -> Abort immediately! 0 ranks torn down (zero split-brain)

[TRT-LLM _executor_loop_overlap / _consume_previous_batch_for_rebalance (ADP Ranks)]
  Every ADP Rank (whether previous_batch is 'processed', 'deferred', or 'absent'):
    1. _flush_pending_transfer_responses()
    2. _process_previous_batch() [if processed] OR _enqueue_responses([]) [if absent/deferred]
    3. _add_kv_cache_events() -> kv_event_allgather (strictly outside divergent branch!)
```

### 3.2 Mooncake Store V2 KV 虚拟地址池的 `cuMemCreate` 边界切分

- **问题根源**：TRT-LLM V2 KV Cache 池通过 CUDA Virtual Memory Management (`cuMemAddressReserve` + 多次 `cuMemCreate` / `cuMemMap`) 构建连续虚拟地址区间，并且滑动窗口层组或 MiniMax-M3 `index-K` 与 `K/V` 常常共享或交错在同一虚拟池中。然而 RDMA 网卡在进行 GPUDirect 注册时要求每个注册区间位于单一物理 `cuMemCreate` 分配内，且同一虚拟范围不可重复注册。
- **三步确定性切分算法**：
  1. `merge_intervals`：将所有 Layer Group 的 `[base, span_end)` 合并为互不重叠的最小覆盖区间（如 `[0x10000000, +48 MiB)` 与 `[+16 MiB, +96 MiB)` 合并为单一 `96 MiB` 区间）。
  2. `allocation_spans`：通过 `cuMemGetAddressRange` 驱动查询回调 `lookup(cursor)`，将 `96 MiB` 虚拟区间按 `32 MiB` `cuMemCreate` 物理块边界切分为 **`3` 个 `32 MiB` 注册片**，并记录内部切点 `cuts = [0x12000000, 0x14000000]`。
  3. `split_at_cuts`：当某个 `4 MiB` KV Page（`[+30 MiB, +34 MiB)`）横跨 `0x12000000` 物理切点时，`PageAddressing.buffers()` 用二分查找 `bisect_right` 将其自动拆分为 **`[2 MiB, 2 MiB]`** 两段连续散列缓冲区提交给 Mooncake `batch_transfer_sync`，并在接收端写入前绑定 `AllocationFence` 确保 GPU Stream 顺序一致。

---

## 四、标准库确定性复现与验证指标（Deterministic Reproduction Metrics）

| 评估维度 | 指标名称 | 确定性复现值 | 验证状态 |
|---|---|---|---|
| SGLang PD+DP 切换 | 单阶段部分失败 vs 两阶段 `prepare_only` 脑裂状态 | `split_brain: True (3/4 torn down)` vs `False (0 torn down)` | `PASS` |
| SGLang PD+DP 切换 | `--enable-dp-attention` 允许 / `local-control-broadcast` 拒绝 | `True` / `True` | `PASS` |
| TRT-LLM ADP 事件 | Overlap Loop 三类异构状态组合死锁消除率 | `1/3 无死锁 -> 3/3 (100%) 无死锁` | `PASS` |
| TRT-LLM ADP 事件 | KV Pool Rebalance (`absent` + `processed`) 死锁消除 | `False -> True` | `PASS` |
| Mooncake Zero-Copy | `96 MiB` 虚拟池合并区间数 / `32 MiB` 物理块切分注册数 | `1` / `3` | `PASS` |
| Mooncake Zero-Copy | 跨界 `4 MiB` Page 切分片段 / 8 卡节点节省 Pinned DRAM | `[2 MiB, 2 MiB]` / `1,024 MiB` | `PASS` |

---

## 五、生产采纳建议（Production Guidance）

1. **SGLang 弹性 PD 池配置**：在开启 `--enable-dp-attention` 的 PD 弹性切换集群中，切勿同时开启 `--enable-dp-attention-local-control-broadcast`（此时各进程仅控制本地 DP Rank，缺乏全局两阶段预检屏障）。
2. **GB200 / GB300 无 `nvidia_peermem` 节点零拷贝部署**：升级至包含 `PR #20108` 的 TRT-LLM 后，Mooncake Connector 会自动检测 `/sys/module/nvidia_peermem` 缺失并设置 `WITH_NVIDIA_PEERMEM=0` 走 Linux Kernel DMA-BUF (`ibv_reg_dmabuf_mr`)，无需回退到高延迟 Host Staging。
