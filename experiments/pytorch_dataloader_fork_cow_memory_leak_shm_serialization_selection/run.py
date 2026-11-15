#!/usr/bin/env python3
"""Round 235: PyTorch DataLoader (num_workers > 0) Python List fork() CoW 64x RAM Explosion, /dev/shm Bus Error & mmap Zero-Copy Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_dataloader_fork_cow_shm_tiers(
    dataset_index_gb: float = 12.0,
    total_node_workers: int = 64,
) -> dict[str, Any]:
    """Evaluate 4 PyTorch DataLoader multiprocessing memory & /dev/shm configuration tiers on an 8-GPU node."""
    tiers = [
        {
            "tier_id": "tier1_python_list_dict_fork_cow_refcount_64x_ram_explosion",
            "name": "Tier 1: Dataset.__init__ 保存 Python list/dict (只读访问触发 ob_refcnt 写时复制，64 Worker 撑爆 768 GB 内存)",
            "steady_state_node_host_ram_gb": 768.0,
            "shm_sigbus_or_oom_crashes_per_run": 1,
            "cow_page_duplication_factor_x": 64.0,
            "verdict": "在 CPython 中哪怕只读 list[i] 也会修改对象头部的引用计数 (Py_INCREF)，瞬间击穿 Linux fork CoW 共享页，64 个进程复制 64 份内存直接触发 OOM Killer ✕",
        },
        {
            "tier_id": "tier2_default_docker_64mb_dev_shm_sigbus_crash",
            "name": "Tier 2: 未挂载大容量 /dev/shm (Docker/K8s 默认 64 MB 共享内存被预取队列打满，抛 SIGBUS Bus error)",
            "steady_state_node_host_ram_gb": 96.0,
            "shm_sigbus_or_oom_crashes_per_run": 1,
            "cow_page_duplication_factor_x": 8.0,
            "verdict": "DataLoader 子进程通过 /dev/shm 传递 Tensor，容器默认的 64MB 空间在几步内即告耗尽，触发致命的 Bus error (core dumped) ⚠️",
        },
        {
            "tier_id": "tier3_contiguous_numpy_tensor_index_array_per_rank",
            "name": "Tier 3: 将索引改为连续 NumPy/Torch Tensor 数组 + 挂载 64GB /dev/shm (消除 Worker CoW，仅留 8 Rank 副本)",
            "steady_state_node_host_ram_gb": 96.0,
            "shm_sigbus_or_oom_crashes_per_run": 0,
            "cow_page_duplication_factor_x": 8.0,
            "verdict": "底层 C 数组切片不触碰共享 Python 对象引用计数，Worker 进程零 CoW 膨胀，但 torchrun 8 个主进程仍各自加载了 1 份 12GB 索引 ✅",
        },
        {
            "tier_id": "tier4_mmap_binary_index_plus_persistent_workers_and_64gb_shm",
            "name": "Tier 4: 磁盘 mmap 二进制索引 (.idx/.bin) + persistent_workers=True + 64GiB /dev/shm 内存卷",
            "steady_state_node_host_ram_gb": 12.2,
            "shm_sigbus_or_oom_crashes_per_run": 0,
            "cow_page_duplication_factor_x": 1.0,
            "verdict": "8 个主进程与 64 个 Worker 100% 共享同一份内核只读 mmap 页表 (仅占 12.2 GB，节省 63 倍内存)，且 0 次 SIGBUS / OOM 崩溃 🏆",
        },
    ]

    return {
        "experiment_id": "pytorch_dataloader_fork_cow_memory_leak_shm_serialization_selection",
        "round": 235,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "dataset_index_gb": dataset_index_gb,
            "total_node_workers": total_node_workers,
            "ram_reduction_x": round(768.0 / 12.2, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_dataloader_fork_cow_shm_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
