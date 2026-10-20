# TensorRT-LLM KVCM V2 原生留存策略（`cold_tier_offload_min_priority=30`）、LMCache 共享 L1 `MemoryOrchestrator` 状态机与 Mooncake TENT `Notify` 结构化应答核验

**语言 / Language**: [中文](trtllm-kvcm-v2-retention-lmcache-l1-orchestrator-mooncake-tent.md) | [English](trtllm-kvcm-v2-retention-lmcache-l1-orchestrator-mooncake-tent.en.md)

> **证据级别 / Evidence Tier**: `implementation`（一手上游提交源码）+ `own-offline`（Python 3.12 标准库确定性验证 `experiments/trtllm_kvcm_v2_retention_lmcache_l1_tent_notify/run.py --check`）。未声明生产多机 RDMA 硬件实测。

---

## 1. 核心架构问题与结论速览

在多级 KV Cache 存储管理（GPU HBM $\to$ Host DRAM $\to$ 共享 L1 / 远端存储）与控制面通知（Notification RPC）中，2026-10-20 的三项上游提交补齐了冷层卸载准入过滤、多进程共享内存租约状态机以及 RPC 投递失败分类的关键契约：

1. **TensorRT-LLM `KVCacheManagerV2` 原生留存优先级与冷层卸载门槛（[`NVIDIA/TensorRT-LLM@41913522c7`](https://github.com/NVIDIA/TensorRT-LLM/commit/41913522c7f9f085c022421fb6183bd7592e076f)）**：
   - **原生块级留存策略**：在 `KVCacheManagerV2`（`kvCache.cpp`、`storageManager.cpp`）中接入 `KvCacheRetentionConfig`（优先级范围 `0..100`，默认 `kPriorityDefault = 35`，支持按 token 区间配置 `(priority, duration_ms)` 并在到期后通过 `refreshRetention` 降级回默认优先级）。
   - **冷层卸载门槛（`cold_tier_offload_min_priority = 30`）**：新增 `kDefaultColdTierOffloadMinPriority = 30`（废弃旧别名 `secondary_offload_min_priority`）。当无引用的 GPU 页面被驱逐时，优先级 `< 30` 的低优页面（如临时思维链草稿、一次性 scratch 块）直接丢弃（drop）而不写入二级冷存储（Host DRAM / Disk），在典型 6 页混合优先级淘汰场景下将 PCIe/NVMe 冷层写放大降低 **`50.0%`**（`1,572,864 -> 786,432 B`）。
   - **队列安全优先级更新与 Helix CP 超块缩放**：`StorageManager::updatePriority` 先持有 `keepAlive = page.sharedFromThis()`，若页面处于驱逐队列则先 `excludeFromEviction(page)` 摘除，修改 `page.mPriority` 后再重新插入目标优先级 LRU 桶并触发 `EventManager::addPriorityUpdated`。在 Helix Context Parallelism（`_has_cp_helix`）下，`_retention_config_for_request` 将 rank 本地物理 token 区间乘以 `_helix_cp_size` 映射至全局 CP super-block 序号；同时 `prefetchReuse(..., isPrefetch=true)` 传入 `claimRetention=false`，防止后台预取覆盖已有留存策略。

2. **LMCache 多进程共享 L1 `MemoryOrchestrator` 权威状态机与幂等重放窗口（[`LMCache/LMCache@258746ce77`](https://github.com/LMCache/LMCache/commit/258746ce7750b085a8cec89d371ac7eaa4adb94d)）**：
   - **`RegionState` 内存分配与生命周期状态机**（`lmcache/v1/memory_orchestrator/state.py`）：作为单个共享 L1 内存区域的唯一元数据权威（KV 字节流不经过此 gRPC 通道），按 2 的幂次 `alignment_bytes` 执行单调 bump 分配（`_round_up(payload_bytes, alignment)`），管理对象状态 `WRITING -> VALID`（或写中止/超时/旧化身退役后转入 `CONSUMED` 并从索引移除）。
   - **全有或全无批量预留与客户端化身隔离（Incarnation Fencing）**：`reserve_write` 对批次中所有缺失键计算对齐后总需求字节数 `needed_bytes`，若 `_allocated_bytes + needed_bytes > capacity_bytes`，整批缺失项统一返回 `OUT_OF_SPACE` 且不分配任何字节（`0` 部分分配泄漏）。当同一 `client_id` 以新的 `client_incarnation` 重新注册时，`_retire(previous)` 自动将该客户端旧化身持有的全部 `WRITING` 对象转为 `CONSUMED` 并释放其全部读租约（`leases`）。
   - **单权威防脑裂与 `_ReplyWindow` 幂等重放**：`server.py` 显式设置 `_GRPC_OPTIONS = (("grpc.so_reuseport", 0),)` 并维护 `<region_id>.marker` 文件（异常重启标记 `reset_required=True`，拒绝一切业务 RPC）；`OrchestratorServicer` 按客户端化身维护容量 `1024` 的 `_ReplyWindow`（记录 `request_id -> (request_sha256, reply_bytes)`），字节一致的重试直接重放缓存响应（`REPLAYED_OK`），同 `request_id` 但负载篡改的请求立即以 `ALREADY_EXISTS` 拒绝。

3. **Mooncake TENT `ControlService::onNotify` 结构化拒收原因与重试分类（[`kvcache-ai/Mooncake@5cf142e344`](https://github.com/kvcache-ai/Mooncake/commit/5cf142e3447435faa9a9f6c846aea592bcfdc263)）**：
   - **问题根因**：在 `mooncake-transfer-engine/tent/src/runtime/control_plane.cpp` 中，旧版 `ControlService::onNotify` 对所有 `Notify` RPC 均返回空响应（`response = ""`），导致发送端 `ControlClient::notify` 即使在接收端尚未注册回调（RPC 服务先于引擎注册回调启动，或停机阶段先清空回调）、JSON 解析失败或回调函数抛异常/返回非零码时，仍误判为 `Status::OK()`（3 类异常场景下产生 `3/3` 静默假阳性投递确认）。
   - **修复机制**：`onNotify` 仅在成功投递时保持 `response` 为空；未注册回调时返回 `"notify callback not registered"`，JSON 解析失败时返回 `"malformed notification: ..."` 前缀，回调返回非零或抛异常时返回 `"notify callback failed: ..."`。`ControlClient::notify` 将 `"malformed notification: "` 映射为不可重试的 `Status::InvalidArgument`，将其余非空错误映射为可重试的 `Status::InternalError`（消除全部 `3/3` 假阳性确认）。

---

## 2. 状态机流转与确定性离线验证结果 (`own-offline`)

通过 `python experiments/trtllm_kvcm_v2_retention_lmcache_l1_tent_notify/run.py --check` 对三项系统状态机进行了确定性建模验证（详见 `experiments/trtllm_kvcm_v2_retention_lmcache_l1_tent_notify/results/summary.json`）：

| 子系统 / 模块 | 关键故障或压力场景 | 修复前行为 (Legacy) | 修复后契约 (Verified Implementation) | 定量验证指标 (`own-offline`) |
|---|---|---|---|---|
| **TRT-LLM `KVCacheManagerV2`** | 6 页无引用 GPU 页面淘汰（3 页优先级 `80/50/35` $\ge 30$，3 页 `25/20/10` $< 30$） | 无 V2 原生优先级过滤，全量卸载 `1,572,864 B` | `< 30` 直接丢弃，仅卸载 `786,432 B`；Helix CP `4x` 区间缩放 `[0, 1024)` | 冷层写入字节 **`-50.0%`**，预取 `claimRetention=False` |
| **LMCache `MemoryOrchestrator`** | 客户端 `incarnation=1` 崩溃遗留 2 个 `WRITING` 对象 + 超容批量写 + 重复 `request_id` | 僵尸 `WRITING` 锁死键、部分分配或 `SO_REUSEPORT` 双权威脑裂 | `incarnation=2` 自动退役 `2` 个 `WRITING` 为 `CONSUMED`；超容整批 `OUT_OF_SPACE` | 幂等重试 `REPLAYED_OK`，篡改负载 `ALREADY_EXISTS` |
| **Mooncake TENT `Notify` RPC** | 回调未注册 / 非法 JSON / 回调返回 `-5` | 全部返回空串 $\to$ `Status::OK()`（`3` 次假阳性投递） | 区分 `InvalidArgument`（不重试）与 `InternalError`（可重试） | 静默假阳性确认从 **`3 -> 0`** |

---

## 3. 面向解决方案架构师的设计练习与可证伪问题

1. **架构师排障设计练习**：
   - 在 PD 分离推理集群中，Decode 节点刚启动 Mooncake TENT RPC Server 但尚未完成 KV Cache 接收回调注册时，Prefill 节点发起了首批 `Notify` 通知。在 `Mooncake@5cf142e344` 修复后，Prefill 端 `ControlClient::notify` 会收到什么状态码？上层调度器应当如何处理？
   - **解答**：接收端 `onNotify` 返回 `"notify callback not registered"`，客户端将其映射为 `Status::InternalError`（而非 `InvalidArgument`），上层发送器应执行指数退避重试直至接收端完成回调注册。
2. **后续可证伪问题**：
   - LMCache `RegionState` 当前采用单调递增的 bump 分配器（`_allocated_bytes` 只增不减，`CONSUMED` 对象的 extent 空间尚未回收复用）。在持续发生写超时或客户端重启的长时间运行集群中，何时必须触发进程级 region epoch 重置，或者后续 PR 如何引入 extent 空闲链表回收？
