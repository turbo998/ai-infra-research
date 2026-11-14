#!/usr/bin/env python3
"""Round 206: NCCL PXN (Proxy by NVLink) Cross-Rail Spine Hairpin Elimination & Hierarchical Rail Routing Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_nccl_pxn_tiers(
    nic_line_rate_gbs: float = 50.0,
) -> dict[str, Any]:
    """Evaluate 4 cross-rail routing tiers in an 8-Rail 400GbE/NDR SU cluster under non-rail-aligned traffic."""
    tiers = [
        {
            "tier_id": "tier1_pxn_disabled_cross_rail_spine_hairpin",
            "name": "Tier 1: 禁用 PXN (NCCL_PXN_DISABLE=1 · 87.5% 跨轨流量强爬 Spine 交换机引发 ECMP 哈希碰撞)",
            "spine_traffic_ratio_pct": 87.5,
            "per_gpu_busbw_gbs": 19.4,
            "p99_collective_tail_ms": 28.5,
            "verdict": "Node A GPU 0 发往 Node B GPU 3 必须从 Leaf S0 爬上 Spine 再下到 Leaf S3，引发 Spine 拥塞与 PFC 死锁，单卡带宽跌至 19.4 GB/s ✕",
        },
        {
            "tier_id": "tier2_pxn_without_gdr_pix_host_bounce_fallback",
            "name": "Tier 2: 开启 PXN 但未打通 PCIe PIX GPUDirect RDMA (退化为 Host CPU 内存弹簧床代理)",
            "spine_traffic_ratio_pct": 0.0,
            "per_gpu_busbw_gbs": 24.5,
            "p99_collective_tail_ms": 19.2,
            "verdict": "虽然消除了 Spine 流量，但代理卡无法通过 GPUDirect RDMA 直读本卡 HBM，被迫回落至 CPU 内存拷贝 ⚠️",
        },
        {
            "tier_id": "tier3_nccl_pxn_enabled_with_gdr_pix_and_p2p_level2",
            "name": "Tier 3: 标准硬件加速 PXN (NCCL_PXN_DISABLE=0 + NCCL_P2P_PXN_LEVEL=2 + GDR PIX)",
            "spine_traffic_ratio_pct": 0.0,
            "per_gpu_busbw_gbs": 44.2,
            "p99_collective_tail_ms": 5.8,
            "verdict": "先走机内 900 GB/s NVLink 到同号代理 GPU，再由同轨 Leaf 交换机一跳直达目标节点，Spine 流量归零，带宽跃升至 44.2 GB/s ✅",
        },
        {
            "tier_id": "tier4_hierarchical_nvlink_rail_aligned_deepep_pxn",
            "name": "Tier 4: 显式分层机内 NVLink 聚合 + 纯同轨 RDMA 直传 (DeepEP / 拓扑感知 Rank 编排 + PXN)",
            "spine_traffic_ratio_pct": 0.0,
            "per_gpu_busbw_gbs": 48.1,
            "p99_collective_tail_ms": 3.4,
            "verdict": "在内核层显式完成机内 NVLink 预聚合后仅发起 Rail i -> Rail i 单跳 RDMA，规避 CPU Proxy 线程瓶颈，跑满 48.1 GB/s (96.2% 线速) 🏆",
        },
    ]

    return {
        "experiment_id": "nccl_pxn_nvlink_proxy_cross_rail_spine_selection",
        "round": 206,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "nic_line_rate_gbs": nic_line_rate_gbs,
            "rails_per_node": 8,
            "pxn_speedup_x": round(48.1 / 19.4, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_nccl_pxn_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
