#!/usr/bin/env python3
"""Round 197: GPUDirect Storage (GDS / cuFile / nvidia-fs), NVMe P2P & io_uring vs POSIX Buffered I/O Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_gds_nvme_tiers(
    nvme_drives_count: int = 8,
    single_gen5_nvme_gbs: float = 14.0,
) -> dict[str, Any]:
    """Evaluate 4 NVMe-to-GPU I/O architectures across 8 PCIe Gen5 NVMe SSDs (112.0 GB/s raw peak)."""
    raw_peak_gbs = nvme_drives_count * single_gen5_nvme_gbs  # 112.0 GB/s

    tiers = [
        {
            "tier_id": "tier1_posix_buffered_page_cache_double_copy",
            "name": "Tier 1: 传统 POSIX Buffered I/O (Linux Page Cache + 用户态拷贝 + cudaMemcpy)",
            "aggregate_nvme_to_gpu_gbs": 18.5,
            "raw_bandwidth_utilization_pct": round((18.5 / raw_peak_gbs) * 100.0, 1),
            "host_dram_traffic_multiplier": 3.0,
            "host_cpu_utilization_pct": 92.0,
            "verdict": "数据在主机 DDR5 内存里进出 3 次并冲刷内核 Page Cache，8 块 Gen5 SSD 仅跑出 18.5 GB/s 且 CPU 打满 92% ✕",
        },
        {
            "tier_id": "tier2_o_direct_io_uring_pinned_host_bounce",
            "name": "Tier 2: O_DIRECT + io_uring 异步轮询 + 锁页内存 (Pinned Host Bounce Buffer)",
            "aggregate_nvme_to_gpu_gbs": 48.0,
            "raw_bandwidth_utilization_pct": round((48.0 / raw_peak_gbs) * 100.0, 1),
            "host_dram_traffic_multiplier": 2.0,
            "host_cpu_utilization_pct": 28.0,
            "verdict": "绕过了内核 Page Cache 拷贝，但仍需先 DMA 进主机内存再拷入显存 (2.0x 内存总线占用)，受限于 CPU 根复合体带宽 ⚠️",
        },
        {
            "tier_id": "tier3_cufile_gds_cpu_root_complex_routed",
            "name": "Tier 3: GPUDirect Storage (cuFile) 跨 CPU Root Complex 路由 (NVMe 挂在 CPU 直连通道)",
            "aggregate_nvme_to_gpu_gbs": 64.5,
            "raw_bandwidth_utilization_pct": round((64.5 / raw_peak_gbs) * 100.0, 1),
            "host_dram_traffic_multiplier": 0.0,
            "host_cpu_utilization_pct": 11.0,
            "verdict": "免除了主机内存 Bounce Buffer，但因 NVMe 插在 CPU 直连槽位而非 GPU PCIe Switch 下，TLP 包仍需穿透 CPU RC ✅",
        },
        {
            "tier_id": "tier4_gds_cufile_pcie_switch_pix_p2p_zero_copy",
            "name": "Tier 4: GPUDirect Storage (nvidia-fs + cuFile 4KB 对齐 + NVMe 与 GPU 同处 PIX Switch)",
            "aggregate_nvme_to_gpu_gbs": 98.5,
            "raw_bandwidth_utilization_pct": round((98.5 / raw_peak_gbs) * 100.0, 1),
            "host_dram_traffic_multiplier": 0.0,
            "host_cpu_utilization_pct": 6.5,
            "verdict": "NVMe 控制器在 PCIe Switch 内部直接向 GPU BAR1 显存发起 P2P DMA，零主机内存流量，狂飙 98.5 GB/s (快 5.32x) 🏆",
        },
    ]

    return {
        "experiment_id": "gpudirect_storage_gds_cufile_nvme_iouring_selection",
        "round": 197,
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload": {
            "nvme_drives_count": nvme_drives_count,
            "single_gen5_nvme_gbs": single_gen5_nvme_gbs,
            "raw_peak_gbs": raw_peak_gbs,
            "posix_gbs": 18.5,
            "gds_pix_gbs": 98.5,
            "throughput_speedup_x": round(98.5 / 18.5, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    results = evaluate_gds_nvme_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
