# Megatron-LM LayerShardedMuon 流式分批 Newton-Schulz 与 TE 256-Token 对齐、Mooncake 自适应 RDMA 拥塞控制 / 多租户配额及 NIXL 即时完成追踪

> **English Version**: [megatron-lsmuon-ns-stream-te-align-mooncake-acc-quota-nixl.en.md](megatron-lsmuon-ns-stream-te-align-mooncake-acc-quota-nixl.en.md)  
> **关联问题 ID**: `megatron-lsmuon-ns-stream-te-align-mooncake-acc-quota-nixl` (`priority: 50`, Phase 41)  
> **关联离线实验**: [`experiments/megatron_lsmuon_stream_mooncake_acc_quota_nixl/run.py`](../../experiments/megatron_lsmuon_stream_mooncake_acc_quota_nixl/run.py) · [`summary.json`](../../experiments/megatron_lsmuon_stream_mooncake_acc_quota_nixl/summary.json)  
> **证据等级**: `implementation` (第一手源码审计) + `own-offline` (Python 标准库确定性建模)

---

## 1. 架构背景与核心问题

在大规模 MoE 训练与分离式 KV Cache 存储/RDMA 传输平面中，`2026-10-21` 上游社区集中演进了五项关键基础设施机制：

1. **Megatron-LM `LayerShardedMuon` 瞬态显存峰值与单 Rank 退化问题 (`#8379`)**：在百亿/千亿参数 MoE 训练中，单个 Home Rank 或单 Rank 通信域（`gtp_remat x tp == 1`，如非 TP 切分的 Replicated 层或单卡测试）往往持有上百个同形状权重矩阵。若一次性构造全部矩阵的 FP32 Nesterov 动量输入并把全部 Newton-Schulz (NS) 正交化结果保存在字典中再集中更新权重，会产生数 GiB 的瞬态显存尖峰；同时此前在单 Rank 域上即使调用了 `set_param_ns_homes()`，也会退回父类 `TensorParallelMuon` 逐参数循环调用未分批 NS，丧失同形状 `torch.stack` + Batched SYRK 的加速。
2. **Megatron-LM Transformer Engine `GroupedLinear` 256-Token 硬件对齐 (`#8184`)**：Blackwell/Hopper CuTeDSL/CUTLASS Grouped GEMM 对每个专家分片行数的对齐粒度要求从早期的环境变量开关演进为原生 `split_alignment = 256` 参数。
3. **Mooncake Transfer Engine 自适应 RDMA 拥塞控制核心 (`#4197`)**：在高并发 KV Cache RDMA 传输中，静态并发窗口无法同时兼顾正常态满带宽与 RoCE/IB RNR (Receiver Not Ready) 拥塞态防雪崩，且网卡故障重连（Generation 切换）时在途反馈信号容易污染新周期窗口。
4. **Mooncake Store 多租户内存配额 `TenantQuotaPolicy` (`#4809`)**：多租户共享内存池在执行同名 Key 覆盖写（Upsert）时，若新副本尚处于 `PROCESSING` 写入态而旧副本已被替换但尚未释放物理内存，单纯按当前元数据计费会导致租户在途双倍占用逃逸配额限制。
5. **NIXL UCX 同步快路径缺失 `nixl::xfer.complete` 追踪标记 (`#2462`)**：当 `postXferReq` 在首次投递时即同步完成（`status == NIXL_SUCCESS`），旧代码直接返回而未发射 `nixl::xfer.complete` NVTX/Trace 标记，导致 `tracing_nsys` 捕获的 50 次传输中丢失了 17 次完成事件（仅记录 33 次异步轮询完成），破坏了 P99 传输延迟分析的配对完备性。

---

## 2. Megatron-LM `LayerShardedMuon` 流式 `_iter_ns` 与 TE `split_alignment=256`

### 2.1 `_iter_ns` 按 Chunk 流式正交化 (`layer_sharded_muon.py:526-574`, `764-775`)

在 `NVIDIA/Megatron-LM` Commit `6c956376e3ef22528413ff35412ad8b938272986` (PR `#8379`) 中，`LayerShardedMuon` 将 `_run_ns` 拆解为生成器 `_iter_ns(self, full_by_k, load=lambda t: t)`：

```python
for ks in by_shape.values():
    for start in range(0, len(ks), self.ns_batch_size):
        chunk = ks[start : start + self.ns_batch_size]
        batched = len(chunk) > 1
        x = [load(full_by_k[k]) for k in chunk]
        x = torch.stack(x) if batched else x[0]
        orth = newton_schulz(
            x, steps=self.num_ns_steps, coefficient_type=self.coefficient_type, use_syrk=self.use_syrk
        )
        scale = get_muon_scale_factor(orth.size(-2), orth.size(-1), mode=self.scale_mode)
        orth.mul_(scale).mul_(self.extra_scale_factor)
        if batched:
            yield from zip(chunk, orth.unbind(0))
        else:
            yield chunk[0], orth
        del orth
```

**三项核心工程收益**：
1. **瞬态显存从 $O(N_{\text{params}})$ 降至 $O(\text{ns\_batch\_size})$**：对 Replicated 参数路径 (`layer_sharded_muon.py:769-775`)，传入 `load=ns_input` 后，仅当某个 Chunk 开始执行时才按需构造该 Chunk 内 `ns_batch_size` 个参数的 FP32 Nesterov 动量张量，产出 `ns_output` 后立即调用 `self._apply_update(params[i], ns_output, lr)` 并 `del ns_output`。对于 $N = 128$ 个 `[2048, 2048]` FP32 矩阵、`ns_batch_size = 8`，峰值瞬态显存从 **4096.0 MiB** 降至 **256.0 MiB**（**降低 16.0x**）。
2. **单 Rank 域保留同形状分批加速**：当 `gtp_remat x tp == 1` 且已调用 `set_param_ns_homes()` 时，全部参数归入 `plan.replicated` 并走 `_iter_ns` 按形状分批执行 Batched SYRK Newton-Schulz（调用次数从 $128$ 次降至 $16$ 次），而不再退回父类 `TensorParallelMuon` 的逐参数未分批循环。
3. **两步原地乘法严格保持位级对齐 (`orth.mul_(scale).mul_(self.extra_scale_factor)`)**：显式分两次执行 FP32 乘法而非预先将 `scale * extra_scale_factor` 合并为单个 Python float，确保与 `TensorParallelMuon` 的浮点舍入逐位一致；且 `tp_complete`（仅跨 `gtp_remat` 切分、在各 TP 列间复制的参数）单独调用一次 `_run_ns` (`layer_sharded_muon.py:845-858`)，防止不同 TP 列因混入各自不同的 TP 分片参数改变 Chunk 划分（`baddbmm` vs `addmm`）而引发跨 TP 副本权重漂移。

### 2.2 Transformer Engine `TE_GROUPED_GEMM_EXPERT_TOKEN_ALIGNMENT = 256` (`#8184`)

在 Commit `45b68f914cf15eae6eaa40117fe94cd6a3972d4c` (PR `#8184`) 中，`megatron/core/extensions/transformer_engine.py` 定义常量 `TE_GROUPED_GEMM_EXPERT_TOKEN_ALIGNMENT = 256`，并通过 `inspect.signature(te.pytorch.ops.GroupedLinear)` 检测 `split_alignment` 参数，在 `TEGroupedMLP`（`experts.py`）、`SharedExpertMLP`（`shared_experts.py`）和 Token Dispatcher（`token_dispatcher.py`）中统一以 256-Token 对齐驱动 CuTeDSL Grouped GEMM，移除了旧版 `NVTE_CUTEDSL_FUSED_GROUPED_MLP` 环境变量分支。

---

## 3. Mooncake 自适应 RDMA 拥塞控制 (`#4197`)、多租户配额 (`#4809`) 与 NIXL 追踪修复 (`#2462`)

### 3.1 Mooncake Adaptive RDMA Congestion Control (`adaptive_congestion_control.cpp`)

在 `kvcache-ai/Mooncake` Commit `3b08b77168870d265e32b6cfeab661f037426b9c` (PR `#4197`) 中，`mooncake::adaptive_congestion_control` 实现了双层（`Device` + `Route`）无锁准入与分代（Generation）隔离状态机：

1. **双域两阶段准入与 Generation 围栏 (`tryAcquire`, `adaptive_congestion_control.cpp:249-317`)**：
   - 先对 `path.device` 执行 `acquireOne`，再对 `path.route` 执行 `acquireOne`；若 Route 拒绝，立即回滚释放 Device 已预留的 `inflight_bytes_`。
   - **空域超大 Slice 放行规则 (`tryReserveBytes`, lines `85-100`)**：判断条件为 `if (current != 0 && (bytes > limit || current > limit - bytes)) return false;`。这意味着当某个域当前 `inflight_bytes_ == 0` 时，即使单个 RDMA 传输 Slice 大小（如 `16 MiB`）超过当前拥塞窗口 `window_bytes_`（如最小窗口 `4 MiB`），仍允许该单个 Slice 通过（防止大 Slice 永久饿死），但在其完成前阻止后续并发 Slice 进入。
   - **Generation 校验**：准入完成前再次核对 `path.device_generation` 与 `path.route_generation` 是否等于当前原子 `generation_`；若期间发生网卡重连重置 (`resetGeneration`)，回滚字节并返回 `Decision::kAvoid` (`AdmissionReason::kStaleGeneration`)。
2. **基于排空时间 (`drainTimeNs`) 与 RNR 的 AIMD 窗口演化 (`observeEpoch`, lines `380-442`)**：
   - 默认配置：`min_window_bytes = 4 MiB`, `max_window_bytes = 64 MiB`, `target_drain_time_ns = 2,000,000 ns (2 ms)`, `high_pressure_epochs = 2`, `low_pressure_epochs = 3`。
   - 排空时间计算：$\text{drain\_ns} = \lfloor \text{backlog\_bytes} \times 10^9 / \text{delivery\_rate\_Bps} \rfloor$。
   - **高压减半 (`PathState::kCongested`)**：若 `receiver_pressure != 0`（注意：RNR `kReceiverPressure` 仅记入 `route` 域而不污染共享 `device` 域，见 lines `176-181`）或 $\text{drain\_ns} > 2\,\text{ms}$，连续达到 `2` 个 Epoch 后窗口减半：$W \leftarrow \max(4\,\text{MiB}, \lfloor W / 2 \rfloor)$。
   - **低压加性增长 (`PathState::kHealthy`)**：若存在成功完成且 $\text{drain\_ns} < 1\,\text{ms}$ (`target_drain_time_ns / 2`)，连续达到 `3` 个 Epoch 后窗口增加 $\Delta = \max(1, \text{min\_window\_bytes} / 4) = 1\,\text{MiB}$：$W \leftarrow \min(64\,\text{MiB}, W + 1\,\text{MiB})$。

### 3.2 Mooncake Store `TenantQuotaPolicy` 双字记账与水位驱逐 (`quota_policy.cpp`)

在 `kvcache-ai/Mooncake` Commit `44c4d9cf4ace3ce2955312a2467b1305d910dd48` (PR `#4809`) 中，`TenantQuotaPolicy` 利用每个 Key 的 `owner_state().policy_words[0..1]` 解决了并发写入与同名 Key 覆盖写（Upsert）的精确配额记账问题：

- **`Held(ctx)` (`policy_words[0]`) 与 `Carried(ctx)` (`policy_words[1]`)**：
  - 写入扩容时，`OnGrow(ctx, bytes)` (`quota_policy.cpp:164-180`) 先向 `TenantQuotaAccount::TryCharge(bytes)` 预扣配额（超限则递增 `tenant_quota_reject` 指标并返回 `ErrorCode::TENANT_QUOTA_EXCEEDED`），并累加 `Held(ctx) += bytes`。
  - 当 Upsert 替换已有对象时，`OnReplace(ctx)` (`lines 182-187`) 将被替换旧对象的内存账单累加至 `Carried(ctx) += MemoryCharge(old_metadata)`。
  - 在释放写锁结算 `OnWriteRelease(ctx)` (`lines 195-222`) 时，只要新对象的内存副本仍处于 `ReplicaStatus::PROCESSING` (`HasProcessingMemoryReplica(metadata) == true`)，`Carried(ctx)` **不会清零**，该 Key 应欠总字节数为：
    $$\text{owed} = \text{MemoryCharge}(\text{new\_metadata}) + \text{Carried}(\text{ctx})$$
    直到新副本写入完成（不再处于 `PROCESSING`），`Carried(ctx)` 才归零并通过 `account.Release(held - owed)` 退还旧对象占用的配额。這彻底堵住了租户通过高频 Upsert 在途副本占用双倍物理内存却只按单份计费的漏洞。
- **双水位 Slab 驱逐 (`OnMaintenance`, `quota_policy.cpp:247-296`)**：当租户使用率 $\text{used\_ratio} = \text{charged} / \text{effective\_quota} > \text{watermark}$ 时，一次性向下驱逐至 $\text{target\_ratio} = \max(0, \text{watermark} - \text{eviction\_ratio})$：
  $$\text{target\_bytes} = \lfloor (\text{used\_ratio} - \text{target\_ratio}) \times \text{effective\_quota\_bytes} \rfloor$$

### 3.3 NIXL `postXferReq` 同步完成追踪修复 (`nixl_agent.cpp`)

在 `ai-dynamo/nixl` Commit `f8248873bd5b8d3609ba1448fa2f623aefb361cf` (PR `#2462`) 中，针对 UCX 小块或 RDMA 立即完成传输在 `postXferReq` 内直接返回 `NIXL_SUCCESS` 导致丢失 `nixl::xfer.complete` 事件的问题，在三处补齐了关联追踪标记：
1. `nixlAgent::postXferReq` 首次投递立即完成 (`status == NIXL_SUCCESS`) 时发射 `NIXL_TRACE_MARK(data->tracer_.get(), "nixl::xfer.complete", nixl::trace::Kind::Metadata)`；
2. `nixlAgent::postXferReq` 重投递（repost）立即完成时发射相同标记；
3. `nixlAgent::releaseXferReq` 在 `req_h->status == NIXL_SUCCESS` 时（包裹在 `NIXL_TRACE_CORRELATION_SCOPE` 内）补发完成标记，同时避免将已取消 (`NIXL_IN_PROG` cancelled) 的请求误标为完成。在作者的 50 次 UCX 传输测试中，`nixl::xfer.complete` 捕获数从 **33 次恢复至完整的 50 次**。

---

## 4. 本仓库离线验证 (`own-offline`)

运行 `python experiments/megatron_lsmuon_stream_mooncake_acc_quota_nixl/run.py --check`，确定性验证了全部四项不变量：
- **`LayerShardedMuon` 流式 `_iter_ns`**：在 128 个 `[2048, 2048]` 矩阵、`ns_batch_size = 8` 下，峰值瞬态显存从 `4096.0 MiB` 降至 `256.0 MiB`（**16.0x 降低**），Batched NS 调用次数为 `16` 次，且 TE Grouped GEMM 对齐常数为 `256`。
- **Mooncake ACC 状态机**：连续 6 个低压周期窗口从 `4.0 MiB` 增长至 `6.0 MiB`，随后 2 个高压周期减半回落至 `4.0 MiB` (`kCongested`)；空域下 `16 MiB` 超大 Slice 获准 (`kAllow`)，非空域下被延迟 (`kDefer`)。
- **Mooncake `TenantQuotaPolicy`**：`400 B` 旧对象被 `500 B` 新对象 Upsert 期间，在途 `PROCESSING` 态严格维持 `owed = 900 B`，完成后归零 `Carried` 并释放 `400 B`；在 `92%` 占用率（`watermark=0.90, eviction_ratio=0.20`）下精确计算驱逐目标为 `220 B`。
- **NIXL `xfer.complete` 配对完备性**：修复后 50 次传输（含 17 次同步立即完成 + 33 次异步完成）实现 `50 / 50` 雙射配对。
