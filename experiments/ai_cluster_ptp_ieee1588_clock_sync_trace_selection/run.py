#!/usr/bin/env python3
"""Round 188: AI Cluster Hardware Clock Synchronization (NTP vs PTP IEEE 1588v2 + PHC + SyncE) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_ptp_clock_sync_tiers(
    allreduce_step_us: float = 250.0,
) -> dict[str, Any]:
    """Evaluate 4 time synchronization tiers for distributed AI straggler attribution and RDMA OWD."""
    tiers = [
        {
            "tier_id": "tier1_standard_ntp_chrony_kernel_socket",
            "name": "Tier 1: 传统 NTP (chrony 内核协议栈时间戳 · 毫秒级漂移)",
            "max_clock_skew_ns": 2400000,  # 2,400,000 ns = 2.4 ms
            "max_clock_skew_us": 2400.0,
            "straggler_attribution_accuracy_pct": 12.5,
            "owd_congestion_telemetry_capable": False,
            "trace_causality_violation_rate_pct": 68.0,
            "verdict": "时钟偏差 (2,400us) 是 AllReduce 耗时 (250us) 的 9.6 倍！Trace 出现“先收后发”因果倒置，无法定位慢节点 ✕",
        },
        {
            "tier_id": "tier2_software_ptp4l_without_switch_bc",
            "name": "Tier 2: 纯软件 PTP (ptp4l 无网卡 PHC 硬件戳 / 交换机无透明钟)",
            "max_clock_skew_ns": 85000,  # 85 us
            "max_clock_skew_us": 85.0,
            "straggler_attribution_accuracy_pct": 54.0,
            "owd_congestion_telemetry_capable": False,
            "trace_causality_violation_rate_pct": 18.5,
            "verdict": "受操作系统中断抖动与 RoCE 交换机排队不对称影响，抖动仍达 85us，难以分辨微秒级内核延迟 ⚠️",
        },
        {
            "tier_id": "tier3_ieee1588_nic_phc_leaf_transparent_clock",
            "name": "Tier 3: IEEE 1588v2 网卡硬件 PHC + ToR 透明时钟 (TC) + phc2sys",
            "max_clock_skew_ns": 450,  # 0.45 us
            "max_clock_skew_us": 0.45,
            "straggler_attribution_accuracy_pct": 96.5,
            "owd_congestion_telemetry_capable": True,
            "trace_causality_violation_rate_pct": 0.2,
            "verdict": "网卡 MAC 层硬件打戳消除 OS 抖动，交换机修正驻留时间，跨机 Trace 对齐达 450ns，精准锁定慢卡 ✅",
        },
        {
            "tier_id": "tier4_synce_plus_ptp_boundary_clock_gpu_timer_sync",
            "name": "Tier 4: SyncE 物理层锁频 + 全网 PTP 边界时钟 (BC) + GPU globaltimer 硬件对齐",
            "max_clock_skew_ns": 35,  # 0.035 us (35 ns)
            "max_clock_skew_us": 0.035,
            "straggler_attribution_accuracy_pct": 99.8,
            "owd_congestion_telemetry_capable": True,
            "trace_causality_violation_rate_pct": 0.0,
            "verdict": "35ns 纳秒级全网同步！支持 RDMA 单向延迟 (OWD) 拥塞控制与万卡 GPU 内核级时间轴零误差拼图 🏆",
        },
    ]

    return {
        "experiment_id": "ai_cluster_ptp_ieee1588_clock_sync_trace_selection",
        "round": 188,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "allreduce_step_us": allreduce_step_us,
            "ntp_skew_us": 2400.0,
            "ptp_bc_skew_ns": 35,
            "precision_improvement_x": round(2400000 / 35, 1),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_ptp_clock_sync_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
