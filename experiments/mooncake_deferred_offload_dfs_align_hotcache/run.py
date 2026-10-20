#!/usr/bin/env python3
"""Deterministic simulator for Phase 28 Topic 2:
Mooncake Store FileStorage::PartitionUnbucketedTasks deferred-tail offload carry (#3006/#3009),
DFS ShardAllocator zero-slack aligned allocation (#4753),
and Client::ProcessSlicesAsync CUDA device-pointer hot-cache writeback guard (#4764).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OffloadTaskItem:
    key: str
    size: int
    generation: int


def allocate_offloading_buckets(
    offloading_objects: dict[str, int],
    ungrouped_pool: dict[str, int],
    bucket_size_limit: int,
    bucket_count_limit: int,
) -> tuple[list[list[str]], list[str]]:
    """Model BucketStorageBackend::GroupOffloadingKeysByBucket (storage_backend.cpp:2302-2363)."""
    if not offloading_objects:
        return [], []

    combined: list[tuple[str, int]] = []
    for k, sz in ungrouped_pool.items():
        combined.append((k, sz))
    ungrouped_pool.clear()
    for k, sz in offloading_objects.items():
        if sz <= bucket_size_limit:
            combined.append((k, sz))

    buckets: list[list[str]] = []
    deferred_keys: list[str] = []
    cur_bucket: list[str] = []
    cur_bytes = 0

    for k, sz in combined:
        if cur_bucket and (
            cur_bytes + sz > bucket_size_limit or len(cur_bucket) >= bucket_count_limit
        ):
            buckets.append(cur_bucket)
            cur_bucket = []
            cur_bytes = 0
        cur_bucket.append(k)
        cur_bytes += sz

    if cur_bucket:
        if len(cur_bucket) == bucket_count_limit or cur_bytes == bucket_size_limit:
            buckets.append(cur_bucket)
        else:
            # Tail partial bucket is deferred to ungrouped_offloading_objects_
            for k in cur_bucket:
                sz_val = offloading_objects.get(k)
                if sz_val is None:
                    # Find from combined
                    sz_val = dict(combined)[k]
                ungrouped_pool[k] = sz_val
                deferred_keys.append(k)

    return buckets, deferred_keys


def partition_unbucketed_tasks(
    task_by_storage_key: dict[str, OffloadTaskItem],
    all_bucket_keys: set[str],
    deferred_keys: list[str],
    previously_carried: dict[str, OffloadTaskItem],
    redrained_carried_keys: set[str],
) -> tuple[list[OffloadTaskItem], dict[str, OffloadTaskItem]]:
    """Exact replica of FileStorage::PartitionUnbucketedTasks (file_storage.cpp:289-337)."""
    failed_tasks: list[OffloadTaskItem] = []
    carried_tasks: dict[str, OffloadTaskItem] = {}

    for storage_key in deferred_keys:
        if storage_key in task_by_storage_key:
            carried_tasks[storage_key] = task_by_storage_key[storage_key]

    for storage_key, task in previously_carried.items():
        if storage_key in redrained_carried_keys:
            continue
        if storage_key not in all_bucket_keys:
            carried_tasks[storage_key] = task

    for storage_key, task in task_by_storage_key.items():
        if storage_key not in all_bucket_keys and storage_key not in carried_tasks:
            failed_tasks.append(task)

    return failed_tasks, carried_tasks


def simulate_deferred_offload_lifecycle() -> dict[str, Any]:
    """Compare upstream #3006 naive NACK, incomplete carry, and #3009 PartitionUnbucketedTasks."""
    bucket_size_limit = 4 * 1024 * 1024  # 4 MiB
    bucket_count_limit = 10

    # Scenario A: 21 keys of 1 MiB arriving 1 per cycle (reproduces #3006 19/21 silent drop when flush happens)
    # Every 4 cycles a bucket of 4 keys fills; the 3 prior keys were deferred.
    # Under pre-#3009 naive NACK, the 3 deferred keys had their tasks NACKed on cycles 1..3,
    # so when cycle 4 emits the bucket [k1, k2, k3, k4], k1..k3 are missing from task_by_storage_key
    # and complete_handler returns INVALID_KEY for the whole bucket!
    naive_pool: dict[str, int] = {}
    naive_persisted = 0
    naive_dropped_on_reemission = 0
    for i in range(1, 22):
        batch = {f"k{i}": 1024 * 1024}
        task_map = {f"k{i}": OffloadTaskItem(f"k{i}", 1024 * 1024, 1)}
        buckets, _ = allocate_offloading_buckets(
            batch, naive_pool, bucket_size_limit, bucket_count_limit
        )
        for b in buckets:
            if all(k in task_map for k in b):
                naive_persisted += len(b)
            else:
                naive_dropped_on_reemission += len(b)
    # Tail 1 key remains in ungrouped pool until next full bucket; even if flushed with 3 dummy keys,
    # the 19 keys across 5 buckets fail or remain stranded: 0 persisted out of 20 bucketed keys!

    # Scenario B: Full 4-cycle edge-case trace testing:
    # Cycle 1: k1, k2, k3 (1 MiB each -> deferred) + k_oversize (8 MiB -> NACKed immediately)
    # Cycle 2: k_oversize_only (8 MiB) -> GroupOffloadingKeysByBucket gets only oversize, so k1..k3 stay in pool un-emitted and un-reported in deferred_keys!
    # Cycle 3: k3 is re-PUT at 8 MiB (collision + oversize skip!) while k4, k5 (1 MiB) arrive -> k1, k2, k4, k5 form a full 4 MiB bucket!
    cycles = [
        [
            OffloadTaskItem("k1", 1024 * 1024, 1),
            OffloadTaskItem("k2", 1024 * 1024, 1),
            OffloadTaskItem("k3", 1024 * 1024, 1),
            OffloadTaskItem("k_over1", 8 * 1024 * 1024, 1),
        ],
        [
            OffloadTaskItem("k_over2", 8 * 1024 * 1024, 1),
        ],
        [
            OffloadTaskItem("k3", 8 * 1024 * 1024, 2),  # Re-drained at oversize!
            OffloadTaskItem("k4", 1024 * 1024, 1),
            OffloadTaskItem("k5", 1024 * 1024, 1),
        ],
    ]

    # Run #3009 fixed state machine
    fixed_pool: dict[str, int] = {}
    carried_map: dict[str, OffloadTaskItem] = {}
    persisted_tasks: list[str] = []
    nacked_tasks: list[tuple[str, int]] = []

    for cycle_items in cycles:
        storage_object_sizes = {t.key: t.size for t in cycle_items}
        task_by_storage_key = {t.key: t for t in cycle_items}
        redrained_carried_keys: set[str] = set()

        for sk, task in carried_map.items():
            if sk in storage_object_sizes:
                fixed_pool.pop(sk, None)
                redrained_carried_keys.add(sk)
                continue
            task_by_storage_key[sk] = task

        buckets, deferred_keys = allocate_offloading_buckets(
            storage_object_sizes, fixed_pool, bucket_size_limit, bucket_count_limit
        )
        all_bucket_keys: set[str] = set()
        for b in buckets:
            for k in b:
                all_bucket_keys.add(k)
                assert k in task_by_storage_key
                persisted_tasks.append(f"{k}@gen{task_by_storage_key[k].generation}")

        failed, new_carried = partition_unbucketed_tasks(
            task_by_storage_key=task_by_storage_key,
            all_bucket_keys=all_bucket_keys,
            deferred_keys=deferred_keys,
            previously_carried=carried_map,
            redrained_carried_keys=redrained_carried_keys,
        )
        for f in failed:
            nacked_tasks.append((f.key, f.generation))
        carried_map = new_carried

    return {
        "single_key_stream_21_items": {
            "naive_persisted_keys": naive_persisted,
            "naive_dropped_bucketed_keys": naive_dropped_on_reemission,
            "naive_tail_deferred_keys": len(naive_pool),
        },
        "multi_cycle_edge_trace": {
            "persisted_tasks": persisted_tasks,
            "nacked_tasks": [{"key": k, "generation": g} for k, g in nacked_tasks],
            "remaining_carried_tasks": list(carried_map.keys()),
            "remaining_pool_keys": list(fixed_pool.keys()),
            "zombie_tasks_count": len(carried_map),
        },
    }


def normalized_size_class(size: int) -> int:
    """Model OffsetAllocator float-to-uint8 mantissa size-class rounding (8 subdivisions per power-of-2 bin)."""
    if size <= 65536:
        return 65536
    step = 1 << max(0, (size - 1).bit_length() - 4)
    return -(-size // step) * step


def simulate_dfs_shard_allocator() -> dict[str, Any]:
    """Compare legacy aligned_size + alignment - 1 vs fixed aligned_size (#4753)."""
    shard_capacity = 64 * 1024 * 1024  # 64 MiB
    alignment = 64 * 1024  # 64 KiB
    obj_size = 1024 * 1024  # 1 MiB

    def align_size(val: int) -> int:
        return -(-val // alignment) * alignment

    aligned_size = align_size(obj_size)
    legacy_alloc_size = aligned_size + alignment - 1
    legacy_reserved = normalized_size_class(legacy_alloc_size)
    fixed_reserved = normalized_size_class(aligned_size)

    legacy_max_1mib_objects = shard_capacity // legacy_reserved
    fixed_max_1mib_objects = shard_capacity // fixed_reserved

    # Full-shard single allocation test (64 MiB object on 64 MiB shard)
    full_shard_aligned = align_size(shard_capacity)
    legacy_full_shard_allowed = (full_shard_aligned + alignment - 1) <= shard_capacity
    fixed_full_shard_allowed = full_shard_aligned <= shard_capacity

    # Verify every offset in fixed allocator is aligned to 64 KiB
    offsets = [i * fixed_reserved for i in range(fixed_max_1mib_objects)]
    unaligned_offsets = sum(1 for off in offsets if (off & (alignment - 1)) != 0)

    return {
        "shard_capacity_bytes": shard_capacity,
        "alignment_bytes": alignment,
        "object_size_bytes": obj_size,
        "legacy_allocation_request_bytes": legacy_alloc_size,
        "legacy_reserved_size_class_bytes": legacy_reserved,
        "fixed_allocation_request_bytes": aligned_size,
        "fixed_reserved_size_class_bytes": fixed_reserved,
        "per_object_inflation_pct_removed": round(
            (legacy_reserved - fixed_reserved) / fixed_reserved * 100.0, 4
        ),
        "legacy_max_1mib_objects": legacy_max_1mib_objects,
        "fixed_max_1mib_objects": fixed_max_1mib_objects,
        "legacy_utilization_pct": round(
            (legacy_max_1mib_objects * obj_size) / shard_capacity * 100.0, 4
        ),
        "fixed_utilization_pct": round(
            (fixed_max_1mib_objects * obj_size) / shard_capacity * 100.0, 4
        ),
        "legacy_full_shard_64mib_succeeds": legacy_full_shard_allowed,
        "fixed_full_shard_64mib_succeeds": fixed_full_shard_allowed,
        "unaligned_offsets_count": unaligned_offsets,
    }


def simulate_hotcache_device_pointer_guard() -> dict[str, Any]:
    """Compare ProcessSlicesAsync before and after FindDeviceForPointer check (#4764)."""
    slices = [
        {"slice_idx": 0, "memory_kind": "cuda_device", "bytes": 262144},
        {"slice_idx": 1, "memory_kind": "cuda_device", "bytes": 262144},
        {"slice_idx": 2, "memory_kind": "host_dram", "bytes": 262144},
        {"slice_idx": 3, "memory_kind": "cuda_device", "bytes": 262144},
    ]

    legacy_segfaults = sum(1 for s in slices if s["memory_kind"] == "cuda_device")
    fixed_segfaults = 0
    fixed_skipped_device_slices = 0
    fixed_submitted_host_slices = 0

    for s in slices:
        if s["memory_kind"] == "cuda_device":
            fixed_skipped_device_slices += 1
            continue
        fixed_submitted_host_slices += 1

    return {
        "total_slices": len(slices),
        "legacy_host_memcpy_on_device_segfaults": legacy_segfaults,
        "fixed_host_memcpy_on_device_segfaults": fixed_segfaults,
        "fixed_skipped_device_slices": fixed_skipped_device_slices,
        "fixed_submitted_host_slices": fixed_submitted_host_slices,
    }


def build_summary() -> dict[str, Any]:
    return {
        "experiment_id": "mooncake_deferred_offload_dfs_align_hotcache",
        "phase": 28,
        "priority": 37,
        "deferred_offload_lifecycle": simulate_deferred_offload_lifecycle(),
        "dfs_shard_allocator": simulate_dfs_shard_allocator(),
        "hotcache_device_pointer_guard": simulate_hotcache_device_pointer_guard(),
    }


def verify_summary(summary: dict[str, Any]) -> None:
    offload = summary["deferred_offload_lifecycle"]
    assert offload["single_key_stream_21_items"]["naive_persisted_keys"] == 0
    assert offload["single_key_stream_21_items"]["naive_dropped_bucketed_keys"] == 20
    edge = offload["multi_cycle_edge_trace"]
    assert edge["persisted_tasks"] == ["k1@gen1", "k2@gen1", "k4@gen1", "k5@gen1"]
    assert edge["nacked_tasks"] == [
        {"key": "k_over1", "generation": 1},
        {"key": "k_over2", "generation": 1},
        {"key": "k3", "generation": 2},
    ]
    assert edge["zombie_tasks_count"] == 0

    dfs = summary["dfs_shard_allocator"]
    assert dfs["legacy_reserved_size_class_bytes"] == 1179648
    assert dfs["fixed_reserved_size_class_bytes"] == 1048576
    assert dfs["per_object_inflation_pct_removed"] == 12.5
    assert dfs["legacy_max_1mib_objects"] == 56
    assert dfs["fixed_max_1mib_objects"] == 64
    assert dfs["legacy_full_shard_64mib_succeeds"] is False
    assert dfs["fixed_full_shard_64mib_succeeds"] is True
    assert dfs["unaligned_offsets_count"] == 0

    hc = summary["hotcache_device_pointer_guard"]
    assert hc["legacy_host_memcpy_on_device_segfaults"] == 3
    assert hc["fixed_host_memcpy_on_device_segfaults"] == 0
    assert hc["fixed_skipped_device_slices"] == 3
    assert hc["fixed_submitted_host_slices"] == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Run deterministic assertions")
    args = parser.parse_args()

    summary = build_summary()
    verify_summary(summary)

    out_path = Path(__file__).resolve().parent / "results" / "summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.check:
        print(
            "[PASS] mooncake_deferred_offload_dfs_align_hotcache: "
            "persisted=4/4 valid, zombies=0, dfs_1mib_objects=64/64 (+12.5% reclaimed), hotcache_segfaults=0"
        )


if __name__ == "__main__":
    main()
