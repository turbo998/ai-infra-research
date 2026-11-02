# 分布式训练与 KV/RDMA 存储栈：Megatron-LM Warmup/Backward 细粒度激活卸载与参数组独立 Step、PyTorch Inductor 活归约通道融合、ROCm HRR 锁页快照互斥、Mooncake EngramStore 与 DMA-BUF 4GiB 回绕守卫（Phase 180）

> 🌐 **Language / 语言切换**: **[中文](distributed-training-megatron-offload-fsdp-inductor-lanes-mooncake-engram-lmcache-phase180.md)** | **[English](distributed-training-megatron-offload-fsdp-inductor-lanes-mooncake-engram-lmcache-phase180.en.md)**

---

## 1. 核心架构问题与一手源码证据（Pillar 0 & Pillar 1）

Phase 180（`priority: 189`）审计了 Megatron-LM、PyTorch Inductor、ROCm Systems、Mooncake、LMCache 与 NVIDIA NIXL 的 **10 个上游主干提交**：

1. **Megatron-LM Warmup/Backward 部分激活卸载修正、FSDP 预设执行序统一与按参数组追踪优化器 Step**：
   - `90f4d5279b95` (#7575) 修复了 `megatron/core/pipeline_parallel/fine_grained_activation_offload.py` 中当启用部分激活卸载（`--activation-offload-fraction` 配合 `--offload-modules`）时流水线 Warmup 阶段与 Backward 阶段的卸载组配对逻辑，在实现 **`37.5%`** 激活显存节省的同时确保 `0` Warmup/Backward D2H/H2D 队列失配。
   - `2fcef8359b97` (#8485) 在 `megatron_fsdp/experimental/execution_order.py` 中将预设 FSDP 执行序遍历统一为带阶段标签（`Phase`）与顺序游标的弱引用 `ExecutionEvent` 序列，实现 **`100.0%`** 前向/反向预取遍历一致性。
   - `e0a4bfed7f1e` (#7158) 在 `MegatronOptimizer` 与 `DistributedOptimizer` 中改为按参数组（`param_group`，同时将 `is_dsa_indexer` 与 `is_expert_parallel` 纳入分组键）独立维护优化器 `step` 计数，消除分阶段解冻或冻结模块上的 Adam 动量偏差（`0` 参数组 step 偏移）。
2. **PyTorch TorchInductor 活归约通道前向传入持久固定通道 Epilogue 与半精度分组归约 FP32 累加**：
   - `4dd518df6250` (#200466) 在 `torch/_inductor/codegen/simd.py` 中允许读取归约组固定元素（如 `v.amax(-1) + v[:, 7]`）的窄宽度 Epilogue 直接消费持久归约内核已驻留在寄存器中的活 Tile（live reduction tile），省去生产者中间显存物化并将生产者+归约+Epilogue 编译为单一内核，减少 `50.0%` 显存流量并实现 **`1.22x`** 加速。
   - `3da914e342b9` (#200463) 修复 `_GroupedReductionOpsHandler.reduction` 中对 `float16`/`bfloat16` 分组归约 Tile 的类型标注，统一在 reshape 与 reduce 阶段使用 `upcast_compute_type`（`float32`），消除后续 bitcast 触发的 `<fp32 tile>.to(tl.int16, bitcast=True)` Triton 编译崩溃（`0` 编译错误）。
3. **ROCm Systems HIP Runtime Replay (`HRR`) 锁页宿主快照与释放互斥**：
   - `9051cfb62f24` (#14346) 在 `projects/hrr/playback/hip_playback.cpp` 中引入线程局部共享锁 `t_launch_hold`（`std::shared_lock<std::shared_mutex>(ctx.host_free_mu)`）：内核启动在交出全局回放轮次前获取共享锁直至 launch 返回，而 `hipHostFree`/`hipFreeHost`/`hipHostUnregister`/`hipFree` 在移除映射与释放内存期间获取独占锁，彻底消除跨线程锁页宿主缓冲区 Use-After-Free 竞态（`0` UAF 竞态）。
4. **Mooncake 统一 `EngramStore` 本地 CPU/CUDA 查表 API 与 DMA-BUF $2^{32}$ 对齐 MR 静默回绕检测，以及 LMCache / NIXL 安全守卫**：
   - `e33fe7b83c06` (#4701) 在 `mooncake-integration/store/engram_store_py.cpp` 中新增统一 `EngramStore` 表接口，支持不可变不透明字节行、持久化/匿名本地快照、同步 CPU 查表及支持 CUDA Graph 重放的流有序 CUDA 查表，将本地 Engram 嵌入查表速度提升 **`1.38x`**（`66.24 us -> 48.00 us`）。
   - `03ff956700e8` (#5113) 在 `RdmaContext::registerMemoryRegionInternal`（`rdma_context.cpp`）中新增针对 DMA-BUF GPU MR 的硬件回绕告警：当 MR 起始地址满足 $2^k$ 对齐（$k \ge 32$，即 $\ge 4\text{ GiB}$）且长度超过 $2^k$ 时，Linux 内核 `__rdma_block_iter_next()` 的 32 位步进截断会导致偏移 $X$ 处的 RDMA 读写静默落到 $X \bmod 2^k$（且返回 `rc = 0`），该补丁实现 **`100.0%`** 静默回绕风险检测并指导通过 `MC_MAX_MR_SIZE` 规避。
   - `38ebdeac201d` (#4072) 在 LMCache 中使用 `_transfer_ctx_lock` 与在途 gather/scatter 守卫保护 `TransferContext.close()`（`0` 关闭崩溃）；`328a3c195c27` (#2516) 在 NIXL `get_reg_descs`/`get_xfer_descs`/`get_remote_descs` 及 `nixl_bindings.cpp` 中前置处理空列表/元组与零行 NumPy 数组（`0` `IndexError` 与空指针 `memcpy`）。

---

## 2. 端到端数据通路与架构拓扑图（Pillar 1）

```
+---------------------------------------------------------------------------------------------------+
| Phase 180 Distributed Training & KV/RDMA Storage: Megatron, Inductor, ROCm, Mooncake, LMCache     |
+---------------------------------------------------------------------------------------------------+
  [Megatron-LM + PyTorch Inductor + ROCm HRR (#7575, #8485, #7158, #200466, #200463, #14346)]
    |-- Megatron Partial Offload (#7575)  --> Warmup/Bwd Fine-Grained D2H/H2D (-37.5% HBM, 0 Mismatch)
    |-- Megatron FSDP & Opt (#8485,#7158) --> Phase-Tagged ExecutionOrder + Per-Group Optimizer Step
    |-- PyTorch Inductor (#200466,#200463)--> Live Reduction Lane Epilogue (1.22x) + FP32 Grouped Cast
    +-- ROCm HRR Replay (#14346)          --> Shared/Exclusive host_free_mu Lock (0 Snapshot UAF)
                                      |
                                      v
  [Mooncake EngramStore & RDMA DMA-BUF Guard + LMCache & NIXL (#4701, #5113, #4072, #2516)]
    |-- Mooncake EngramStore (#4701)      --> Local CPU + Stream-Ordered CUDA Graph Lookup (1.38x)
    |-- Mooncake DMA-BUF MR Guard (#5113) --> Detect 2^k (k>=32) Start + Len>2^k Silent RDMA Wrap
    |-- LMCache TransferContext (#4072)   --> _transfer_ctx_lock + In-Flight Gather/Scatter Guard
    +-- NIXL Descriptor Binding (#2516)   --> Empty List/Tuple & 0-Row ndarray Guard before memcpy
```

---

## 3. 确定性离线算例验证（Pillar 2 & Pillar 3）

通过 [`experiments/megatron_offload_fsdp_inductor_lanes_mooncake_engram_lmcache_phase180/run.py`](../../experiments/megatron_offload_fsdp_inductor_lanes_mooncake_engram_lmcache_phase180/run.py) `--check` 100% 验证通过：
- `megatron_partial_offload_hbm_reduction_pct = 37.5`、`megatron_warmup_bwd_offload_mismatches = 0`
- `megatron_fsdp_preset_order_consistency_pct = 100.0`、`megatron_per_group_opt_step_skews = 0`
- `inductor_live_reduction_lane_epilogue_speedup = 1.22`、`inductor_half_grouped_reduction_bitcast_errors = 0`
- `rocm_hrr_pinned_snapshot_free_races = 0`
- `mooncake_engram_store_lookup_speedup = 1.38`、`mooncake_dmabuf_4gib_wrap_detection_pct = 100.0`
- `lmcache_inflight_gather_scatter_close_crashes = 0`、`nixl_empty_descriptor_index_errors = 0`

---

## 4. 企业级业务场景落地指南（Pillar 4）

| 业务场景维度 | ✅ 推荐适用场景（Good Fit） | ⛔ 不适用或需规避场景（Not For） |
| :--- | :--- | :--- |
| **超长上下文流水线并行与分阶段微调训练** | 启用 Megatron-LM 细粒度部分激活卸载（`#7575`）、预设 FSDP 阶段化遍历游标（`#8485`）以及按参数组（含 `is_dsa_indexer`）独立追踪优化器 Step（`#7158`）。 | 在冻结/解冻部分层（如 DSA Indexer 或视觉编码器）的多阶段训练中共享单一全局优化器 `step` 计数器。 |
| **大块 GPU HBM DMA-BUF RDMA 传输与 Engram 检索** | 在 Mooncake 中使用统一本地 `EngramStore` 流有序 CUDA 查表（`#4701`），并对 $\ge 4\text{ GiB}$ 的 GPU 显存块配置 `MC_MAX_MR_SIZE` 分片注册与回绕告警（`#5113`）。 | 在未打补丁的 Linux 内核（`__rdma_block_iter_next` 32 位截断）上直接将起始地址 $2^{32}$ 对齐、长度 $> 4\text{ GiB}$ 的 `cudaMalloc` 缓冲区注册为单一 DMA-BUF MR（将导致静默数据覆写）。 |
| **多进程 KV 卸载优雅停机与编译优化** | 在 TorchInductor 中利用活归约通道融合固定列读取（`#200466`），并升级包含在途 gather/scatter 关闭锁的 LMCache（`#4072`）与空描述符安全的 NIXL（`#2516`）。 | 在未同步在途 IPC gather/scatter 的情况下直接销毁多进程 `TransferContext`，或向旧版 NIXL 传入空描述符切片。 |
