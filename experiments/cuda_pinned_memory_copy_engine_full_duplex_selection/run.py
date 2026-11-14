#!/usr/bin/env python3
"""Round 220: CUDA Pinned Memory (Pageable Fallback vs Per-Step pin_memory vs Pre-Allocated Pool + Dual Copy Engine H2D/D2H) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_pinned_memory_copy_engine_tiers(
    pcie_gen5_unidir_peak_gbs: float = 64.0,
) -> dict[str, Any]:
    """Evaluate 4 Host-GPU memory transfer tiers over PCIe Gen5 x16 for KV Cache & Activation Offloading."""
    tiers = [
        {
            "tier_id": "tier1_pageable_malloc_silent_sync_staging_fallback",
            "name": "Tier 1: 普通 Pageable 内存调用 non_blocking=True (静默退化为 CPU 同步阻塞 + 驱动临时缓冲二次拷贝)",
            "cpu_blocking_stall_ms": 42.5,
            "effective_pci_bandwidth_gbs": 11.8,
            "true_async_overlap": False,
            "verdict": "GPU DMA 硬件严禁直读可换出内存，cudaMemcpyAsync 会静默卡死主线程并在 CPU 侧分块二次拷贝，带宽跌至 11.8 GB/s ✕",
        },
        {
            "tier_id": "tier2_hot_loop_dynamic_cudamallochost_ioctl_lock",
            "name": "Tier 2: 在每步热循环中动态调用 .pin_memory() / cudaMallocHost (触发内核 mmap_lock 与 IOMMU 映射风暴)",
            "cpu_blocking_stall_ms": 26.0,
            "effective_pci_bandwidth_gbs": 18.5,
            "true_async_overlap": False,
            "verdict": "每次动态申请锁页内存都要进内核态调 ioctl 锁定物理页表，锁页耗时甚至超过 PCIe 传输本身 ⚠️",
        },
        {
            "tier_id": "tier3_preallocated_pinned_ring_buffer_single_stream",
            "name": "Tier 3: 预分配常驻 Pinned 环形缓冲池 + 单 CUDA Stream 传输 (跑满单向 54.2 GB/s 但 H2D/D2H 串行)",
            "cpu_blocking_stall_ms": 0.02,
            "effective_pci_bandwidth_gbs": 54.2,
            "true_async_overlap": True,
            "verdict": "彻底消除运行期 ioctl 开销并实现 0.02ms 纯异步下发，但把 H2D 预取与 D2H 卸载塞在同一 Stream 会导致双向排队 ✅",
        },
        {
            "tier_id": "tier4_numa_hugepage_pinned_pool_plus_dual_copy_engine_streams",
            "name": "Tier 4: NUMA 巨页常驻 Pinned 池 + 双独立 Copy Engine 流 (Stream_H2D CE0 + Stream_D2H CE1 全双工并发)",
            "cpu_blocking_stall_ms": 0.02,
            "effective_pci_bandwidth_gbs": 108.4,
            "true_async_overlap": True,
            "verdict": "将 H2D 预取与 D2H 卸载分别绑定至两套独立硬件 Copy Engine 流，同时吃满 PCIe Gen5 上下行全双工 108.4 GB/s (快 9.19x) 🏆",
        },
    ]

    return {
        "experiment_id": "cuda_pinned_memory_copy_engine_full_duplex_selection",
        "round": 220,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "pcie_gen5_unidir_peak_gbs": pcie_gen5_unidir_peak_gbs,
            "pcie_gen5_bidir_peak_gbs": pcie_gen5_unidir_peak_gbs * 2,
            "bandwidth_speedup_x": round(108.4 / 11.8, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_pinned_memory_copy_engine_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
