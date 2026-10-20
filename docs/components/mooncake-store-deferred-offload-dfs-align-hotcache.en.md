# Mooncake Store Deferred Offload Task Carry, DFS Shard Zero-Slack Alignment & Hot-Cache CUDA Device-Pointer Guard

> **中文版本**: [mooncake-store-deferred-offload-dfs-align-hotcache.md](mooncake-store-deferred-offload-dfs-align-hotcache.md) | **Back to Index**: [README.en.md](../../README.en.md)

## 1. Problem Statement & Architectural Context

In the distributed KV cache storage layer (`kvcache-ai/Mooncake` Mooncake Store), three production bugs across DRAM-to-SSD bucketed offloading, Distributed File System (DFS) shard allocation, and zero-copy client reads directly impacted **KV persistence hit rate, SSD/DFS capacity utilization, and process memory safety**:

1. **Silent `~12%` (`19/21` in single-key streams) Offload Key Loss on Bucket Tail Deferral (`kvcache-ai/Mooncake` `#3006` / `#3009`)**:
   - `BucketStorageBackend::GroupOffloadingKeysByBucket` packs objects into full buckets for SSD writes and defers the remaining tail partial bucket into `ungrouped_offloading_objects_` for a subsequent cycle.
   - Previously, `FileStorage::OffloadObjects` treated every key absent from `all_bucket_keys` in the current cycle as permanently skipped and immediately NACKed its `OffloadTaskItem` (`data_size = -1`) to the master. When a later cycle re-emitted those deferred keys from `ungrouped_offloading_objects_` inside a full bucket, their tasks were missing from `task_by_storage_key`, causing `complete_handler` to fail or silently drop the re-emitted keys.
2. **DFS `ShardAllocator::Allocate` `12.5%` Capacity Inflation from Redundant `alignment_ - 1` Slack (`kvcache-ai/Mooncake` `#4753`)**:
   - `ShardAllocator::Allocate` previously requested `allocation_size = aligned_size + alignment_ - 1` so it could post-align `handle->address()`.
   - Because each shard allocator starts at offset `0` and only receives multiples of the power-of-two `alignment_` (e.g., `64 KiB`), `OffsetAllocator` size-class rounding, splitting, and coalescing already preserve alignment. Adding `alignment_ - 1` pushed a `1 MiB` (`1,048,576 B`) KV block to `1,114,111 B`, which `normalizedAllocationSize` rounded up to the **`1.125 MiB` (`1,179,648 B`)** size class—wasting **`12.5%`** of shard capacity and causing single full-shard (`64 MiB`) allocations to fail with `NO_AVAILABLE_HANDLE`.
3. **`Client::ProcessSlicesAsync` Host `std::memcpy` `SIGSEGV` on CUDA Device Destinations (`kvcache-ai/Mooncake` `#4764`)**:
   - When `batch_get_into` read remote slices directly into CUDA device buffers, `ProcessSlicesAsync` unconditionally passed `slices[i]` to `LocalHotCacheHandler::SubmitPutTask`, which performed a host `std::memcpy` from the CUDA pointer and faulted with `SIGSEGV`.

---

## 2. First-Hand Code Evidence & Implementation Mechanics

### 2.1 `FileStorage::PartitionUnbucketedTasks` Three-Invariant State Machine (`20196fc6f7`)

In `mooncake-store/include/file_storage.h` (blob `ae873da8d5`), `mooncake-store/src/file_storage.cpp` (blob `6ba0a61163`, lines `289–458`), and `mooncake-store/src/storage_backend.cpp` (blob `ff6eed2ab4`, lines `2278–2363`), Mooncake enforces three invariants:

1. **Explicit Deferral Carry**: `GroupOffloadingKeysByBucket` reports tail keys added to `ungrouped_offloading_objects_` via `deferred_keys`; `PartitionUnbucketedTasks` moves their `OffloadTaskItem` entries into `carried_tasks` (`deferred_task_by_storage_key_`) rather than NACKing them.
2. **Retention Across Empty / All-Skipped Cycles**: If a cycle passes where all newly drained keys exceed the bucket limit (so the ungrouped pool is neither emitted nor re-reported in `deferred_keys`), `PartitionUnbucketedTasks` retains un-emitted entries from `previously_carried`.
3. **Re-drain Collision Scrubbing (`RemoveUngroupedOffloadingObject`)**: If a carried key is re-PUT and re-drained before re-emission (`storage_object_sizes.find(storage_key) != end()`), `OffloadObjects` calls `bucket_backend->RemoveUngroupedOffloadingObject(storage_key)` to evict the stale-sized pooled copy and records the key in `redrained_carried_keys` so `PartitionUnbucketedTasks` does not retain the stale task as a zombie or suppress a NACK if the fresh oversize task is skipped.

### 2.2 DFS `ShardAllocator::Allocate` Zero-Slack Aligned Allocation (`7d07515095`)

In `mooncake-store/src/storage/distributed/shard_allocator.cpp` (blob `f433f67114`, lines `261–298`), `ShardAllocator::Allocate` passes `aligned_size` directly to `normalizedAllocationSize(aligned_size)` and `allocate(aligned_size)`, followed by an explicit invariant guard `if ((alloc_offset & (alignment_ - 1)) != 0) return tl::make_unexpected(ErrorCode::INTERNAL_ERROR);`.

### 2.3 `Client::ProcessSlicesAsync` CUDA Device-Pointer Guard (`67df8c414a`)

In `mooncake-store/src/client_service.cpp` (blob `fab103e035`, lines `5626–5650`), `ProcessSlicesAsync` queries `device::GetAcceleratorRegistry().RuntimeAccelerators().FindDeviceForPointer(slices[i].ptr)` and skips `hot_cache_handler_->SubmitPutTask` whenever the destination pointer resides in accelerator memory.

---

## 3. Deterministic Offline Experiment & Quantitative Findings

Verified via [`experiments/mooncake_deferred_offload_dfs_align_hotcache/run.py`](../../experiments/mooncake_deferred_offload_dfs_align_hotcache/run.py) (`own-offline`):

| Evaluation Dimension | Pre-Fix Baseline | Verified Implementation | Architectural Impact |
|---|---|---|---|
| **21-Key `1 MiB` Single-Key Stream (`bucket=4 MiB`)** | Deferred tasks NACKed early; **`20/20` bucketed keys dropped** | `PartitionUnbucketedTasks` carries tasks across cycles; **`0` drops** | Eliminates `#3006` silent SSD offload loss |
| **3-Cycle Edge Trace (All-Oversize Cycle + Re-PUT Collision)** | Drops `k1..k3` or retains zombie task & suppresses `k3@gen2` NACK | Persists `k1,k2,k4,k5`, NACKs `k_over1,k_over2,k3@gen2`, **`0` zombies** | Closes all deferral, empty-cycle, and re-PUT collision edges |
| **DFS `64 MiB` Shard with `1 MiB` Blocks (`align=64 KiB`)** | `1.125 MiB`/obj, fits **`56` objects** (`87.5%`), `64 MiB` object fails | **`1.000 MiB`/obj**, fits **`64` objects** (**`100.0%`**), `64 MiB` succeeds | Reclaims **`12.5%`** shard capacity with `0` unaligned offsets |
| **`batch_get_into` Mixed Slices (`3` CUDA + `1` Host)** | **`3` `SIGSEGV` faults** on host `memcpy` from CUDA pointers | Skips `3` CUDA slices, caches `1` host slice (**`0` faults**) | Safe coexistence of GPUDirect zero-copy reads and host hot cache |

---

## 4. Architect Takeaways & Falsifiable Questions

1. **Co-Design Data Batching Pools with Task Metadata Lifecycles**: Whenever a storage backend defers partial batches across RPC/drain cycles, the caller must maintain a three-way partition (`bucketed`, `carried`, `failed/NACKed`) with explicit collision eviction when a deferred key is overwritten.
2. **Power-of-Two Allocators Starting at Zero Need No Alignment Slack**: Adding `alignment - 1` to an already-aligned power-of-two allocation size crosses size-class boundaries and inflates storage consumption by `12.5%+` without improving alignment.
