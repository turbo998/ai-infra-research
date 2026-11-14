#!/usr/bin/env python3
"""Round 209: 400G/800G Optical Link RS(544,514) FEC Histogram (Bin 0..15), Gray-Failure Detection & Auto-Drain Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_optical_fec_tiers(
    cluster_gpus: int = 1024,
) -> dict[str, Any]:
    """Evaluate 4 optical link telemetry and FEC gray-failure remediation tiers on a 1,024-GPU 400G/800G cluster."""
    tiers = [
        {
            "tier_id": "tier1_binary_link_up_down_only",
            "name": "Tier 1: 仅监控端口 Link Up/Down 状态 (对 Pre-FEC 劣化与灰故障丢包完全失明)",
            "max_tolerated_fec_bin": 15,
            "detection_lead_time_s": 0.0,
            "effective_allreduce_busbw_gbs": 18.3,
            "cluster_goodput_pct": 38.0,
            "verdict": "单根脏污光纤处于 Link Up 但每秒产生数十个 >15 符号不可纠错码字，引发全集群 RDMA 重传雪崩，吞吐暴跌 62% ✕",
        },
        {
            "tier_id": "tier2_reactive_rdma_retransmit_counter_alert",
            "name": "Tier 2: 事后被动监控 RDMA 重传计数器 (packet_seq_err 告警后人工排查)",
            "max_tolerated_fec_bin": 15,
            "detection_lead_time_s": -2700.0,
            "effective_allreduce_busbw_gbs": 29.5,
            "cluster_goodput_pct": 61.2,
            "verdict": "等到重传激增时千卡训练已卡顿数十分钟，人工逐跳定位劣质光模块平均耗时 45 分钟 ⚠️",
        },
        {
            "tier_id": "tier3_pre_fec_ber_threshold_and_uncorrectable_gate",
            "name": "Tier 3: 持续采集 Pre-FEC BER (>1e-5) 与 Uncorrectable FEC == 0 准入闸门",
            "max_tolerated_fec_bin": 11,
            "detection_lead_time_s": 300.0,
            "effective_allreduce_busbw_gbs": 46.4,
            "cluster_goodput_pct": 95.8,
            "verdict": "在 BER 逼近 2.4e-4 断崖前提前 5 分钟拦截劣质链路，但对突发脉冲噪声仍可能漏过零星丢包 ✅",
        },
        {
            "tier_id": "tier4_rs544_fec_tail_bin12_15_plus_cmis_dom_auto_drain",
            "name": "Tier 4: RS(544,514) FEC 尾部直方图 (Bin 12..15 == 0) + CMIS 光功率/SNR 预测性 BGP/OpenSM 秒级摘除",
            "max_tolerated_fec_bin": 7,
            "detection_lead_time_s": 86400.0,
            "effective_allreduce_busbw_gbs": 48.4,
            "cluster_goodput_pct": 99.4,
            "verdict": "在发生哪怕 1 个丢包前 24 小时，通过监控 FEC Bin 12..15 零计数与 CMIS Rx 光功率衰减自动摘除灰故障链路，全集群稳跑 48.4 GB/s 🏆",
        },
    ]

    return {
        "experiment_id": "ai_cluster_optical_fec_rs544_ber_histogram_selection",
        "round": 209,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "cluster_gpus": cluster_gpus,
            "fec_codeword": "RS(544, 514) PAM4",
            "max_correctable_symbols": 15,
            "goodput_recovery_x": round(48.4 / 18.3, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_optical_fec_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
