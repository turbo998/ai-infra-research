#!/usr/bin/env python3
"""Round 262: 10,000-GPU InfiniBand OpenSM Heavy Sweep & Link-Flap LFT Storm vs SHIELD + FLID + Flap Quarantine Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_infiniband_opensm_heavy_sweep_tiers(
    cluster_gpus: int = 10240,
) -> dict[str, Any]:
    """Evaluate 4 InfiniBand Subnet Manager (OpenSM/UFM) routing & link-flap resilience architectures."""
    tiers = [
        {
            "tier_id": "tier1_legacy_static_ftree_without_flap_suppression_global_heavy_sweep",
            "name": "Tier 1: 传统静态 ftree/updn 路由 + 无闪断抑制 (1 个光模块闪断触发全网 Heavy Sweep 重写 LFT，卡死 4,200ms)",
            "failover_convergence_ms": 4200.0,
            "vl15_smp_mad_packets_per_flap": 185000,
            "training_step_degradation_pct": 68.0,
            "verdict": "单端口 Trap 128 迫使中央 OpenSM 重扫万卡拓扑并下发全网交换机 LFT 线性转发表，期间路由震荡拖死全局训练 ✕",
        },
        {
            "tier_id": "tier2_centralized_adaptive_routing_without_switch_shield",
            "name": "Tier 2: 开启自适应路由 (AR)，但未开启交换机硬件 SHIELD 自愈 (仍需等待中央 OpenSM 850ms 改表剔除坏口)",
            "failover_convergence_ms": 850.0,
            "vl15_smp_mad_packets_per_flap": 64000,
            "training_step_degradation_pct": 24.0,
            "verdict": "虽具备多路径组，但端口 Down 时仍需上报 OpenSM 走控制面剔除端口，850ms 窗口内持续黑洞丢包触发重传 ⚠️",
        },
        {
            "tier_id": "tier3_switch_asic_shield_ar_without_port_flap_quarantine",
            "name": "Tier 3: 开启 Quantum-2 硬件 SHIELD + AR (Down 时 0.5ms 芯片级切换，但端口反复 Up 仍触发重扫 320ms)",
            "failover_convergence_ms": 320.0,
            "vl15_smp_mad_packets_per_flap": 28000,
            "training_step_degradation_pct": 8.5,
            "verdict": "断连瞬间由交换机 ASIC 硬件自动屏蔽坏口，但接触不良的光模块每隔几秒恢复 Up 时仍会惊动 OpenSM 重扫 ✅",
        },
        {
            "tier_id": "tier4_shield_ar_plus_flid_hierarchical_routing_and_flap_quarantine",
            "name": "Tier 4: 硬件 SHIELD (<0.5ms 毫秒级自愈) + FLID 聚合路由 (LFT 缩减 12x) + 闪断端口自动隔离 (Flap Dampening)",
            "failover_convergence_ms": 0.4,
            "vl15_smp_mad_packets_per_flap": 1200,
            "training_step_degradation_pct": 0.2,
            "verdict": "ASIC 硬件 0.4ms 瞬时绕行 + FLID 免除跨 Pod LFT 重刷 + 5 分钟闪断 3 次自动隔离坏光模块，训练零感知 🏆",
        },
    ]

    return {
        "experiment_id": "ai_cluster_infiniband_subnet_manager_opensm_lid_reroute_flap_selection",
        "round": 262,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "cluster_gpus": cluster_gpus,
            "fabric_speed": "400G NDR Quantum-2",
            "convergence_speedup_x": round(4200.0 / 0.4, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_infiniband_opensm_heavy_sweep_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
