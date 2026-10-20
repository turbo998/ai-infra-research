# Mooncake Store 延迟卸载任务跨周期保活、DFS 分片零裕量对齐与热缓存 CUDA 设备指针防护深挖

> **English Version**: [mooncake-store-deferred-offload-dfs-align-hotcache.en.md](mooncake-store-deferred-offload-dfs-align-hotcache.en.md) | **返回索引 / Back to README**: [README.md](../../README.md)

## 1. 问题定义与工程背景 (Problem Statement & Architectural Context)

在分布式 KV Cache 存储层（`kvcache-ai/Mooncake` Mooncake Store）的内存-SSD 二级卸载（Tiered Offloading）、分布式文件系统（DFS）大块分配以及客户端零拷贝读取路径中，存在三类直接影响 **KV 持久命中率、SSD/DFS 空间利用率与进程内存安全** 的生产缺陷：

1. **分桶卸载尾部延迟导致的 `~12%`（单流测试 `19/21`）静默丢键（`kvcache-ai/Mooncake` `#3006` / `#3009`）**：
   - `BucketStorageBackend::GroupOffloadingKeysByBucket` 将内存对象聚合为满桶写入 SSD；未凑满一桶的尾部对象被暂存进 `ungrouped_offloading_objects_` 池，等待后续周期合并成桶。
   - 然而旧版 `FileStorage::OffloadObjects` 将所有本周期未进入 `all_bucket_keys` 的键一律视为“被后端跳过（如超大对象）”并立即向 Master 发送 `data_size = -1` NACK！当下一周期 `ungrouped_offloading_objects_` 将这些延迟键重新吐出（re-emit）进某个桶时，当前的 `task_by_storage_key` 中已不存在对应的 `OffloadTaskItem`，导致 `complete_handler` 在查找任务时失败或将其静默丢弃（上游实测约 `12%` 池对象、单键流测试中 `20` 个分桶键全部在重发射时因缺任务而丢失）。
2. **DFS `ShardAllocator::Allocate` 冗余 `alignment_ - 1` 头部裕量造成 `12.5%` 容量膨胀（`kvcache-ai/Mooncake` `#4753`）**：
   - 在 `mooncake-store/src/storage/distributed/shard_allocator.cpp` 中，旧版分配器为应对“返回地址未对齐”而请求 `allocation_size = aligned_size + alignment_ - 1`，再调用 `AlignSize(handle->address())`。
   - 实际上，每个分片分配器均从偏移 `0` 起步，且仅接收 `alignment_`（2 的幂，如 `64 KiB`）整数倍的请求，其底层 `OffsetAllocator` 的 size-class 向上取整、分裂与合并均保持 2 的幂对齐！冗余的 `+ (alignment_ - 1)` 使原本恰好为 `1 MiB`（`1,048,576 B`）的 KV 块变成 `1,114,111 B`，被 `OffsetAllocator` 向上取整到下一个 size class **`1.125 MiB`（`1,179,648 B`）**，凭空浪费 **`12.5%`** 的 DFS 分片容量，且导致单个与分片同尺寸（如 `64 MiB`）的对象因超出 `shard_capacity_` 而直接报错 `NO_AVAILABLE_HANDLE`。
3. **`Client::ProcessSlicesAsync` 对 CUDA 显存目标切片执行宿主 `std::memcpy` 引发 `SIGSEGV`（`kvcache-ai/Mooncake` `#4764`）**：
   - 当客户端通过 `batch_get_into` 将远端 KV 切片直接拉取到 GPU 显存缓冲区（CUDA device pointer）时，`ProcessSlicesAsync` 仍试图将这些非本地切片提交给 `LocalHotCacheHandler::SubmitPutTask`，后者使用宿主 CPU `std::memcpy` 读取 `slices[i].ptr`，立即触发段错误（`SIGSEGV`）。

---

## 2. 一手源码证据与实现机制 (First-Hand Code Evidence & Implementation Mechanics)

### 2.1 `FileStorage::PartitionUnbucketedTasks` 三不变量状态机 (`20196fc6f7`)

在 `mooncake-store/include/file_storage.h`（blob `ae873da8d5`）、`mooncake-store/src/file_storage.cpp`（blob `6ba0a61163`，lines `289–458`）与 `mooncake-store/src/storage_backend.cpp`（blob `ff6eed2ab4`，lines `2278–2363`）中，Mooncake 建立了完整的跨周期任务保活与碰撞清理状态机：

```cpp
void FileStorage::PartitionUnbucketedTasks(
    const std::unordered_map<std::string, OffloadTaskItem>& task_by_storage_key,
    const std::unordered_set<std::string>& all_bucket_keys,
    const std::vector<std::string>& deferred_keys,
    const std::unordered_map<std::string, OffloadTaskItem>& previously_carried,
    const std::unordered_set<std::string>& redrained_carried_keys,
    std::vector<OffloadTaskItem>& failed_tasks,
    std::unordered_map<std::string, OffloadTaskItem>& carried_tasks)
```

该纯函数与 `FileStorage::OffloadObjects` 共同维护三条严密不变量：
1. **显式延迟上报不变量（Deferred Carry）**：`GroupOffloadingKeysByBucket` 在将尾部未满桶对象放入 `ungrouped_offloading_objects_` 时，通过 `deferred_keys->push_back(bucket_object.first)` 显式返回延迟键列表；`PartitionUnbucketedTasks` 将其放入 `carried_tasks`（存入成员变量 `deferred_task_by_storage_key_`），禁止对其发送 NACK。
2. **空输入/全超大周期保留不变量（Un-emitted Retention）**：若下一周期传入的键全部因超大被后端跳过（或合并前有效输入为空），`ungrouped_offloading_objects_` 不会发生移动，因此这些键既不在 `all_bucket_keys` 中，也不会再次出现在本周期的 `deferred_keys` 中。`PartitionUnbucketedTasks` 遍历 `previously_carried`，将所有未进入 `all_bucket_keys`（且未发生重写入碰撞）的旧任务继续保留在 `carried_tasks` 中。
3. **重写入碰撞剔除不变量（Re-drain Collision Scrubbing）**：若某个正处于延迟池中的键在尚未写出到 SSD 前被客户端重新 `PUT` 并再次从 Master 队列 drain 出来（`storage_object_sizes.find(storage_key) != end()`），`OffloadObjects` 立即调用 `bucket_backend->RemoveUngroupedOffloadingObject(storage_key)` 物理删去池中带有旧尺寸的副本，并将该键记入 `redrained_carried_keys`。这样，若新 `PUT` 的对象因尺寸超限被后端跳过，旧 carried 任务不会被保留为僵尸任务（Zombie Task），且新任务能正常进入 `failed_tasks` 发送 NACK 释放源副本引用计数。

### 2.2 DFS `ShardAllocator::Allocate` 零裕量对齐分配 (`7d07515095`)

在 `mooncake-store/src/storage/distributed/shard_allocator.cpp`（blob `f433f67114`，lines `261–298`）中：

```cpp
const uint64_t aligned_size = AlignSize(size);
if (aligned_size > shard_capacity_) {
    return tl::make_unexpected(ErrorCode::NO_AVAILABLE_HANDLE);
}
...
const uint64_t reserved_bytes = shard.allocator->normalizedAllocationSize(aligned_size);
auto handle = shard.allocator->allocate(aligned_size);
if (!handle) continue;

const uint64_t alloc_offset = handle->address();
if ((alloc_offset & (alignment_ - 1)) != 0) {
    LOG(ERROR) << "DFS shard allocator returned unaligned offset "
               << alloc_offset << " for alignment " << alignment_;
    return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);
}
```

- 移除 `+ alignment_ - 1` 后，`1 MiB` 对象（`aligned_size = 1,048,576 B`）直接命中 `1,048,576 B` size class（而非 `1,179,648 B`），并在返回时通过位运算 `(alloc_offset & (alignment_ - 1)) != 0` 做防御性对齐校验。

### 2.3 `Client::ProcessSlicesAsync` CUDA 设备指针热缓存回写拦截 (`67df8c414a`)

在 `mooncake-store/src/client_service.cpp`（blob `fab103e035`，lines `5626–5650`）中：

```cpp
auto runtime_accelerator = device::GetAcceleratorRegistry().RuntimeAccelerators();
size_t device_dest_slices = 0;
for (size_t i = 0; i < slices.size(); ++i) {
    if (runtime_accelerator.FindDeviceForPointer(slices[i].ptr) != nullptr) {
        ++device_dest_slices;
        continue;
    }
    if (!hot_cache_handler_->SubmitPutTask(key, slices[i])) { ... }
}
```

---

## 3. 合成离线实验与量化结论 (Deterministic Offline Experiment & Quantitative Findings)

通过标准库确定性算例 [`experiments/mooncake_deferred_offload_dfs_align_hotcache/run.py`](../../experiments/mooncake_deferred_offload_dfs_align_hotcache/run.py)（`own-offline`）验证：

| 评估维度 | 修复前基线 (Pre-Fix Baseline) | 修复后实现 (Verified Implementation) | 架构收益 / 关键变化 |
|---|---|---|---|
| **21 个 `1 MiB` 单键流分桶卸载 (`bucket=4 MiB`)** | 延迟键被立即 NACK，重发射时 **`20/20` 分桶键因缺任务丢弃** | `PartitionUnbucketedTasks` 跨周期携带任务，**`0` 丢键** | 根除 `#3006` 尾部延迟导致的 `~12%`–`90%+` 静默丢键 |
| **3 周期复合边界（全超大周期 + `k3` 重写入超大碰撞）** | 空周期丢失 `k1..k3`，或碰撞残留僵尸任务并吞没 `k3@gen2` NACK | 落盘 `k1,k2,k4,k5`，精确 NACK `k_over1,k_over2,k3@gen2`，**僵尸任务 `0`** | 完整闭环 `deferred_keys`、`previously_carried` 与 `redrained_carried_keys` |
| **DFS `64 MiB` 分片分配 `1 MiB` 对象 (`align=64 KiB`)** | 占 `1.125 MiB`/个，仅容纳 **`56` 个** (`87.5%` 利用率)，`64 MiB` 单对象失败 | 占 **`1.000 MiB`/个**，容纳 **`64` 个** (**`100.0%` 利用率**)，`64 MiB` 单对象成功 | 消除 **`12.5%`** 大小分级膨胀，且全部分配偏移 `64 KiB` 对齐 (`0` 未对齐) |
| **`batch_get_into` 混合切片热缓存回写 (`3` GPU + `1` CPU)** | 对 `3` 个 CUDA 设备指针执行宿主 `memcpy`，触发 **`3` 次 `SIGSEGV`** | `FindDeviceForPointer` 跳过 `3` 个 GPU 切片，仅缓存 `1` 个宿主切片 (**`0` 崩溃**) | 保障 GPUDirect / CUDA 目标缓冲零拷贝读取与本地热缓存安全共存 |

---

## 4. 解决方案架构师设计准则与可证伪问题 (Architect Takeaways & Falsifiable Questions)

1. **存储后端聚合池与控制面任务生命周期必须同构**：当底层存储引擎（如 `BucketStorageBackend`）引入异步攒批（Ungrouped Pool）时，只保存 `key + size` 会丢失控制面任务上下文（如 `tenant_id`、源副本引用计数回调）；必须在调用方显式区分 **Skipped（立即 NACK）** 与 **Deferred（Carry 至发射完成）**，并在 Key 重写碰撞时同步清理聚合池与 Carry 表。
2. **基于 2 的幂次 Size-Class 分配器无需加 `alignment - 1` 头部裕量**：只要基地址为 `0` 且请求尺寸已按 $2^a$ 对齐，任何保持 $2^a$ 整除性的伙伴/浮点尾数分箱分配器返回的偏移天然满足 `(offset & (alignment - 1)) == 0`；盲目叠加 `alignment - 1` 会跨越 2 的幂边界，触发高达 `12.5%`–`25%` 的内部碎片惩罚。
