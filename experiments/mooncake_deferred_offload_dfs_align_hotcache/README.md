# Experiment: Mooncake Store Deferred Offload Task Carry, DFS Shard Zero-Slack Alignment & Hot-Cache Device Guard

## Purpose (研究目的 / Bilingual Overview)
- **中文**：复现并量化验证 `kvcache-ai/Mooncake` 在 `FileStorage::OffloadObjects` 与 `PartitionUnbucketedTasks` 中对分桶延迟尾部对象（`deferred_keys`）、空输入周期保留（`previously_carried`）与重写入碰撞剔除（`redrained_carried_keys` / `RemoveUngroupedOffloadingObject`）的状态机不变量（`#3006` / `#3009`），分布式文件系统 `ShardAllocator::Allocate` 移除冗余 `alignment_ - 1` 头部裕量（消除 `12.5%` 大小分级膨胀，`#4753`），以及 `Client::ProcessSlicesAsync` 通过 `FindDeviceForPointer` 跳过 CUDA 显存切片的热缓存宿主 `memcpy` 段错误（`#4764`）。
- **English**: Deterministic Python 3.12 standard-library simulator verifying `kvcache-ai/Mooncake` `FileStorage::PartitionUnbucketedTasks` deferred-tail task carry and re-PUT collision scrubbing (`#3006`/`#3009`), DFS `ShardAllocator::Allocate` zero-slack power-of-two alignment (`#4753`), and `Client::ProcessSlicesAsync` CUDA device-pointer hot-cache writeback guard (`#4764`).

## Run & Check
```bash
python experiments/mooncake_deferred_offload_dfs_align_hotcache/run.py --check
```
