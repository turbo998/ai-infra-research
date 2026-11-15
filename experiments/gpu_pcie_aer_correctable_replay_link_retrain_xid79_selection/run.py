#!/usr/bin/env python3
"""Round 223: AI Server PCIe Gen5 AER Correctable Replay Storms, Silent Link Downgrade & XID 79 Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_pcie_aer_replay_xid79_tiers(
    pcie_gen5_x16_nominal_gbs: float = 54.2,
) -> dict[str, Any]:
    """Evaluate 4 PCIe Gen5 health monitoring, link-downgrade detection & XID 79 prevention tiers."""
    tiers = [
        {
            "tier_id": "tier1_unmonitored_aer_replay_storm_and_syslog_flood",
            "name": "Tier 1: 默认内核 AER 刷屏或粗暴 pci=noaer 屏蔽 (放任链路层 NAK 重传风暴与 XID 79 掉卡)",
            "effective_pcie_bandwidth_gbs": 14.2,
            "unplanned_xid79_job_crashes_per_month": 14,
            "silent_link_downgrade_caught": False,
            "verdict": "PCIe 链路层每次 CRC 校验失败都会触发硬件 NAK 重传，不仅将 GPUDirect RDMA 带宽拖垮至 14.2 GB/s，且连续 4 次重传失败即引爆 XID 79 掉卡崩溃 ✕",
        },
        {
            "tier_id": "tier2_ltssm_silent_link_speed_width_downgrade_straggler",
            "name": "Tier 2: 仅监控 GPU 在线状态而漏检 LTSSM 静默降速降宽 (Gen5 x16 退化为 Gen3 x4 隐形慢卡)",
            "effective_pcie_bandwidth_gbs": 3.8,
            "unplanned_xid79_job_crashes_per_month": 6,
            "silent_link_downgrade_caught": False,
            "verdict": "当物理插槽或 Retimer 信号劣化时，PCIe 状态机自动降级至 Gen3 x4 (3.8 GB/s) 保活，nvidia-smi 仍显示 GPU 存活，导致千卡训练步耗时翻倍 ⚠️",
        },
        {
            "tier_id": "tier3_reactive_xid79_node_drain_and_bmc_cold_power_cycle",
            "name": "Tier 3: 事后被动捕获 XID 79 + 自动摘除节点并执行 BMC 冷断电重启 (無法拦截已发生的训练中断)",
            "effective_pcie_bandwidth_gbs": 48.5,
            "unplanned_xid79_job_crashes_per_month": 5,
            "silent_link_downgrade_caught": True,
            "verdict": "虽清楚 XID 79 无法靠ソフト reboot 恢复而必须调 ipmitool power cycle 冷重启，但仍属于掉卡崩溃后的事后补救 ✅",
        },
        {
            "tier_id": "tier4_proactive_dcgm_replay_slope_plus_lnksta_gen5x16_gate",
            "name": "Tier 4: DCGM 硬件重传斜率预警 (>50/min) + lspci LnkSta Gen5x16 强校验 + 任务前自动隔离",
            "effective_pcie_bandwidth_gbs": 54.2,
            "unplanned_xid79_job_crashes_per_month": 0,
            "silent_link_downgrade_caught": True,
            "verdict": "在 Replay Number Rollover 触发 XID 79 之前提前捕捉重传率陡增与 LnkSta 降级，将月均 XID 79 掉卡中断清零 (0 次) 并稳守 54.2 GB/s 满速 🏆",
        },
    ]

    return {
        "experiment_id": "gpu_pcie_aer_correctable_replay_link_retrain_xid79_selection",
        "round": 223,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "pcie_gen5_x16_nominal_gbs": pcie_gen5_x16_nominal_gbs,
            "downgraded_gen3_x4_gbs": 3.8,
            "bandwidth_recovery_x": round(54.2 / 3.8, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_pcie_aer_replay_xid79_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
