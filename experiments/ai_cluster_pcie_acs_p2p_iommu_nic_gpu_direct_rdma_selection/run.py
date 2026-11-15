#!/usr/bin/env python3
"""Round 256: AI Server PCIe ACS (Access Control Services) & IOMMU GPUDirect RDMA Root Complex Detour Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_pcie_acs_iommu_gdr_tiers() -> dict[str, Any]:
    """Evaluate 4 PCIe ACS & IOMMU GPUDirect RDMA routing configurations on 8x400G GPU servers."""
    tiers = [
        {
            "tier_id": "tier1_bios_acs_enabled_plus_strict_iommu_root_complex_hairpin",
            "name": "Tier 1: BIOS 默认开启 PCIe ACS (ReqRedir+ UpstreamFwd+) + Strict IOMMU (P2P 被迫绕行 CPU Root Complex 暴跌至 9.4 GB/s)",
            "per_nic_gdr_busbw_gbps": 9.4,
            "cpu_root_complex_hairpin_traffic_pct": 100.0,
            "iotlb_miss_stall_us": 18.5,
            "verdict": "明明网卡和 GPU 插在同一颗 PCIe Switch 下 (PIX)，ACS 却强迫每个 RDMA 包上送 CPU Root Complex 查 IOMMU 页表再折返 ✕",
        },
        {
            "tier_id": "tier2_acs_disabled_but_strict_iommu_without_passthrough",
            "name": "Tier 2: 关闭 PCIe Switch ACS，但内核未加 iommu=pt (严格 DMA 重映射触发 IOTLB 抖动或 GDR 退化至 Host Bounce)",
            "per_nic_gdr_busbw_gbps": 18.5,
            "cpu_root_complex_hairpin_traffic_pct": 45.0,
            "iotlb_miss_stall_us": 9.2,
            "verdict": "虽然交换芯片放行了 P2P，但非透传 IOMMU 导致 BAR1 地址转换频繁未命中或迫使 NCCL 回退走主机内存中转 ⚠️",
        },
        {
            "tier_id": "tier3_iommu_pt_but_partial_acs_disable_missing_switch_downstream_ports",
            "name": "Tier 3: 配置 iommu=pt 但仅关闭顶层桥 ACS (漏关 PEX 交换芯片内部下游虚拟桥 0604，部分流量仍绕行)",
            "per_nic_gdr_busbw_gbps": 28.0,
            "cpu_root_complex_hairpin_traffic_pct": 30.0,
            "iotlb_miss_stall_us": 1.2,
            "verdict": "只对 Root Port 跑了 setpci 而漏掉了 PCIe Switch 内部的下行端口 (Downstream Ports)，P2P 仍被半途拦截 ✅",
        },
        {
            "tier_id": "tier4_iommu_pt_plus_all_pcie_bridge_setpci_acs_zero_and_peermem",
            "name": "Tier 4: 裸金属 iommu=pt + 全量 PCIe 桥 (0604) setpci ECAP_ACS+0x6.w=0000 + nvidia_peermem 直通",
            "per_nic_gdr_busbw_gbps": 47.5,
            "cpu_root_complex_hairpin_traffic_pct": 0.0,
            "iotlb_miss_stall_us": 0.2,
            "verdict": "100% 流量在 PCIe Switch 内部端口间 150ns 硬件直转 (0% 冲击 CPU Root Complex)，单卡 400G 跑满 47.5 GB/s (提速 5.05x) 🏆",
        },
    ]

    return {
        "experiment_id": "ai_cluster_pcie_acs_p2p_iommu_nic_gpu_direct_rdma_selection",
        "round": 256,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "nic_speed_gbps": 400,
            "topology": "PIX (GPU + ConnectX-7 under PCIe Gen5 Switch)",
            "speedup_vs_acs_enabled_x": round(47.5 / 9.4, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_pcie_acs_iommu_gdr_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
