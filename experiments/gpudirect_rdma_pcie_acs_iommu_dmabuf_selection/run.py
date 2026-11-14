#!/usr/bin/env python3
"""Round 189: GPUDirect RDMA PCIe ACS Redirect, IOMMU IOTLB Thrashing & DMABUF vs nvidia-peermem Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_gpudirect_pcie_acs_tiers(
    nic_line_rate_gbps: int = 400,
) -> dict[str, Any]:
    """Evaluate 4 PCIe / IOMMU / GPUDirect RDMA configurations on a 400Gbps (50 GB/s) NIC-GPU pair."""
    line_rate_gbs = nic_line_rate_gbps / 8.0  # 50.0 GB/s

    tiers = [
        {
            "tier_id": "tier1_host_bounce_buffer_no_gdr",
            "name": "Tier 1: 未加载 GPUDirect 模块 (Host Pinned DRAM Bounce Buffer 两次拷贝)",
            "nccl_busbw_gbs": 9.6,
            "line_rate_utilization_pct": round((9.6 / line_rate_gbs) * 100.0, 1),
            "rdma_p99_latency_us": 28.5,
            "cpu_root_complex_traffic_multiplier": 2.0,
            "verdict": "GPU 显存需先拷回主机内存再由网卡 DMA 发出，PCIe 带宽腰斩且打爆 CPU Root Complex (仅 9.6 GB/s) ✕",
        },
        {
            "tier_id": "tier2_gdr_with_bios_acs_enabled_and_iommu_strict",
            "name": "Tier 2: 开启 GPUDirect 但 BIOS 未关 ACS (ReqRedir+) 且 IOMMU Strict 翻译",
            "nccl_busbw_gbs": 14.5,
            "line_rate_utilization_pct": round((14.5 / line_rate_gbs) * 100.0, 1),
            "rdma_p99_latency_us": 19.2,
            "cpu_root_complex_traffic_multiplier": 1.0,
            "verdict": "PCIe Switch 强制把 P2P 包上送 CPU Root Complex 绕一圈，叠加 141GB BAR1 触发 IOMMU IOTLB 颠簸 ⚠️",
        },
        {
            "tier_id": "tier3_acs_disabled_iommu_pt_nvidia_peermem",
            "name": "Tier 3: 禁用 PCIe ACS + iommu=pt 直通 + 传统 nvidia-peermem 内核模块",
            "nccl_busbw_gbs": 46.8,
            "line_rate_utilization_pct": round((46.8 / line_rate_gbs) * 100.0, 1),
            "rdma_p99_latency_us": 3.4,
            "cpu_root_complex_traffic_multiplier": 0.0,
            "verdict": "流量 100% 在 PCIe Switch (PIX) 内部直通零 CPU 绕行，但依赖外部符号挂钩模块 nvidia-peermem ✅",
        },
        {
            "tier_id": "tier4_pix_acs_off_iommu_pt_kernel_dmabuf",
            "name": "Tier 4: PIX 拓扑 + ACS 关闭 + iommu=pt + Linux 内核原生 DMABUF (NCCL_DMABUF_ENABLE=1)",
            "nccl_busbw_gbs": 48.4,
            "line_rate_utilization_pct": round((48.4 / line_rate_gbs) * 100.0, 1),
            "rdma_p99_latency_us": 2.6,
            "cpu_root_complex_traffic_multiplier": 0.0,
            "verdict": "免第三方 peermem 内核模块！基于 Linux 5.12+ 原生 dma-buf 句柄直驱 BAR1，跑满 48.4 GB/s (96.8% 线速) 🏆",
        },
    ]

    return {
        "experiment_id": "gpudirect_rdma_pcie_acs_iommu_dmabuf_selection",
        "round": 189,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "nic_line_rate_gbps": nic_line_rate_gbps,
            "line_rate_gbs": line_rate_gbs,
            "acs_on_busbw_gbs": 14.5,
            "dmabuf_pix_busbw_gbs": 48.4,
            "busbw_speedup_x": round(48.4 / 9.6, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_gpudirect_pcie_acs_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
