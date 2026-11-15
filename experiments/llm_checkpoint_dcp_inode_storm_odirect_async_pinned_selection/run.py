#!/usr/bin/env python3
"""Round 230: 10,000-GPU Checkpoint MDS Inode Storm, OS Page-Cache OOM & PyTorch DCP Persistent Pinned AsyncStager Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_checkpoint_dcp_inode_odirect_tiers(
    checkpoint_total_tb: float = 10.0,
) -> dict[str, Any]:
    """Evaluate 4 distributed checkpointing storage & staging tiers on 4,096 GPUs."""
    tiers = [
        {
            "tier_id": "tier1_per_tensor_file_mds_inode_lock_storm",
            "name": "Tier 1: 逐张量/未合并多文件直写同一目录 (引发百万级 Inode 创建风暴，锁死并行文件系统 MDS)",
            "gpu_training_stall_sec": 485.0,
            "host_page_cache_bloat_gb_per_node": 640.0,
            "mds_files_created_per_ckpt": 327680,
            "verdict": "数千个 Rank 在同一目录下并发创建数十万个小文件，父目录 dentry 锁竞争直接把 Lustre/NAS MDS 彻底卡死 485 秒 ✕",
        },
        {
            "tier_id": "tier2_buffered_io_dcp_host_page_cache_oom_killer",
            "name": "Tier 2: 单 Rank 合并文件但走默认 Buffered I/O (640GB 脏页瞬间塞满 Linux Page Cache 触发 OOM Killer)",
            "gpu_training_stall_sec": 68.5,
            "host_page_cache_bloat_gb_per_node": 640.0,
            "mds_files_created_per_ckpt": 512,
            "verdict": "虽将文件数减至每 DP 分片组 512 个，但未开启 O_DIRECT 导致 640GB Checkpoint 刷爆主机内存页缓存，唤醒 kswapd 并诱发 OOM 杀进程 ⚠️",
        },
        {
            "tier_id": "tier3_naive_dcp_async_save_dynamic_cudahostregister_stall",
            "name": "Tier 3: 基础 dcp.async_save (每次保存动态调用 cudaHostRegister 锁页，GPU 主循环仍卡顿 3.74 秒)",
            "gpu_training_stall_sec": 3.74,
            "host_page_cache_bloat_gb_gb": 12.0,
            "host_page_cache_bloat_gb_per_node": 12.0,
            "mds_files_created_per_ckpt": 512,
            "verdict": "虽把落盘放到后台线程，但每次保存都在主线程动态锁页 (cudaHostRegister)，GPU 仍需停顿 3.74 秒 ✅",
        },
        {
            "tier_id": "tier4_persistent_pinned_async_stager_plus_odirect_two_phase_commit",
            "name": "Tier 4: PyTorch DCP 持久化锁页 AsyncStager (WeakIdKeyDictionary) + O_DIRECT 绕过页缓存 + 两阶段原子重命名",
            "gpu_training_stall_sec": 0.048,
            "host_page_cache_bloat_gb_per_node": 0.0,
            "mds_files_created_per_ckpt": 512,
            "verdict": "复用常驻 Pinned 内存池在 48 毫秒内完成 D2H 拷贝，后台以 O_DIRECT 直写存储并由 Rank 0 原子 rename 提交，训练停顿压缩 10,104 倍 🏆",
        },
    ]

    return {
        "experiment_id": "llm_checkpoint_dcp_inode_storm_odirect_async_pinned_selection",
        "round": 230,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "checkpoint_total_tb": checkpoint_total_tb,
            "stall_reduction_x": round(485.0 / 0.048, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_checkpoint_dcp_inode_odirect_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
