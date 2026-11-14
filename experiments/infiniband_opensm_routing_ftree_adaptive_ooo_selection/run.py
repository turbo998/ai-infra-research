#!/usr/bin/env python3
"""Round 218: InfiniBand OpenSM Routing Engine (minhop vs ftree vs updn vs ar_ftree/ar_updn + OOO + Fast Recovery) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_infiniband_opensm_routing_tiers(
    line_rate_gbs: float = 50.0,
) -> dict[str, Any]:
    """Evaluate 4 InfiniBand OpenSM/UFM routing engine tiers on an NDR 400G Fat-Tree AI cluster."""
    tiers = [
        {
            "tier_id": "tier1_default_minhop_static_lft_credit_loop",
            "name": "Tier 1: 默认 minhop 静态路由 (无 Fat-Tree 负载均衡 · 静态 LFT 拥塞且易触发 Credit-Loop 死锁)",
            "credit_loop_deadlock_free": False,
            "link_failover_latency_ms": 2500.0,
            "alltoall_busbw_gbs": 24.8,
            "verdict": "仅按最短跳数填静态 LFT 表，多条重流扎堆同一根 Spine 上行链路且断链重算时可能产生环路死锁，AlltoAll 仅 24.8 GB/s ✕",
        },
        {
            "tier_id": "tier2_static_ftree_brittle_fallback_to_minhop",
            "name": "Tier 2: 纯静态 ftree 路由 (单根线缆插错或断链导致拓扑非对称时静默退化回 minhop)",
            "credit_loop_deadlock_free": False,
            "link_failover_latency_ms": 1800.0,
            "alltoall_busbw_gbs": 36.2,
            "verdict": "静态 DLID 映射无法应对 MoE 动态多对一突发，且一旦断纤破坏对称性会悄悄回退至危险的 minhop ⚠️",
        },
        {
            "tier_id": "tier3_deadlock_free_ftree_updn_chain",
            "name": "Tier 3: 防死锁路由链 ftree,updn (严格遵循 Up*/Down* 定理彻底免疫 Credit-Loop 环路死锁)",
            "credit_loop_deadlock_free": True,
            "link_failover_latency_ms": 650.0,
            "alltoall_busbw_gbs": 41.5,
            "verdict": "当 ftree 因断链失效时平滑接管至 updn，从图论上根除先下后上的缓冲区死锁，但仍受限于静态 LFT 表项 ✅",
        },
        {
            "tier_id": "tier4_ar_ftree_ar_updn_plus_rnic_ooo_and_fast_recovery",
            "name": "Tier 4: 硬件自适应路由 (ar_ftree,ar_updn) + 网卡硬件乱序重排 (RNIC OOO) + 交换机 <1ms Fast Recovery",
            "credit_loop_deadlock_free": True,
            "link_failover_latency_ms": 0.8,
            "alltoall_busbw_gbs": 49.2,
            "verdict": "Quantum-2 交换机按实时队列深度逐包动态选路，ConnectX-7 硬件直接处理乱序 RDMA 包，断链 0.8ms 自愈，跑满 49.2 GB/s (98.4% 线速) 🏆",
        },
    ]

    return {
        "experiment_id": "infiniband_opensm_routing_ftree_adaptive_ooo_selection",
        "round": 218,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "line_rate_gbs": line_rate_gbs,
            "busbw_speedup_x": round(49.2 / 24.8, 2),
            "failover_speedup_x": round(2500.0 / 0.8, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_infiniband_opensm_routing_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
