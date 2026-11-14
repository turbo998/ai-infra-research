#!/usr/bin/env python3
"""Round 203: Linux HugePages (4KB vs THP vs 2MB vs 1GB hugetlbfs), GPU MMU TLB & RDMA NIC MTT Cache Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_hugepage_tlb_tiers(
    kv_pool_gb: int = 512,
) -> dict[str, Any]:
    """Evaluate 4 OS memory page size configurations for a 512GB Host DRAM KV-Cache & RDMA pool."""
    pool_bytes = kv_pool_gb * (1024**3)

    tiers = [
        {
            "tier_id": "tier1_standard_4kb_os_pages",
            "name": "Tier 1: 默认 4KB 标准物理页 (512GB 碎裂为 1.34 亿个页表项)",
            "page_size_kb": 4,
            "total_page_table_entries": pool_bytes // 4096,  # 134,217,728
            "mr_pin_registration_time_s": 48.0,
            "random_kv_dma_bandwidth_gbs": 19.8,
            "p99_tail_stall_ms": 14.5,
            "verdict": "1.34 亿个 4KB 页彻底撑爆网卡 MTT 缓存与 GPU IOTLB，随机读 KV 块时 99.9% 触发四级页表回溯，带宽跌至 19.8 GB/s ✕",
        },
        {
            "tier_id": "tier2_transparent_hugepages_thp_always_khugepaged",
            "name": "Tier 2: 透明大页 THP=always (后台 khugepaged 动态碎片整理锁内存)",
            "page_size_kb": 2048,
            "total_page_table_entries": pool_bytes // (2 * 1024 * 1024),
            "mr_pin_registration_time_s": 3.2,
            "random_kv_dma_bandwidth_gbs": 42.5,
            "p99_tail_stall_ms": 95.0,
            "verdict": "平均吞吐虽提升至 42.5 GB/s，但内存碎片化后内核 khugepaged 后台拼页会锁死内存分配长达 95ms，引发推理毛刺 ⚠️",
        },
        {
            "tier_id": "tier3_explicit_2mb_hugetlbfs_pool",
            "name": "Tier 3: 显式预分配 2MB HugePages (hugetlbfs · 26.2 万个页表项)",
            "page_size_kb": 2048,
            "total_page_table_entries": pool_bytes // (2 * 1024 * 1024),  # 262,144
            "mr_pin_registration_time_s": 0.45,
            "random_kv_dma_bandwidth_gbs": 49.2,
            "p99_tail_stall_ms": 1.2,
            "verdict": "页表项减少 512 倍且无后台整理停顿，RDMA 注册缩短至 0.45s，随机 KV 块搬运恢复至 49.2 GB/s ✅",
        },
        {
            "tier_id": "tier4_boot_1gb_hugepages_plus_grace_64kb_ats",
            "name": "Tier 4: 启动预留 1GB 巨页 (仅 512 个页表项!) + Grace ARM64 64KB 基础页 + ATS",
            "page_size_kb": 1048576,
            "total_page_table_entries": pool_bytes // (1024**3),  # 512
            "mr_pin_registration_time_s": 0.04,
            "random_kv_dma_bandwidth_gbs": 54.2,
            "p99_tail_stall_ms": 0.3,
            "verdict": "512GB 内存池仅需 512 个 1GB 页表项！100% 常驻网卡 MTT 与 GPU TLB 缓存，零页表遍历，狂飙 54.2 GB/s (快 2.74x) 🏆",
        },
    ]

    return {
        "experiment_id": "linux_hugepages_1gb_gpu_tlb_rdma_mtt_selection",
        "round": 203,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "kv_pool_gb": kv_pool_gb,
            "pages_4kb": 134217728,
            "pages_1gb": 512,
            "dma_speedup_x": round(54.2 / 19.8, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_hugepage_tlb_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
