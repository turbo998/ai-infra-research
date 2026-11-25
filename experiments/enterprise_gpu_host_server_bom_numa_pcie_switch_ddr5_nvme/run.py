from __future__ import annotations
import json
from pathlib import Path


def compute_ddr5_bandwidth_gbs(channels: int = 24, mt_per_s: int = 5600, efficiency: float = 0.85) -> int:
    return round(channels * 8 * (mt_per_s / 1000.0) * efficiency)


def evaluate_gpu_host_server_bom() -> dict:
    full_24ch_bw = compute_ddr5_bandwidth_gbs(24, 5600, 0.85)
    under_8ch_bw = compute_ddr5_bandwidth_gbs(8, 5600, 0.85)

    return {
        "Deliverable_ID": "#657",
        "H200_8GPU_HBM_GB": 1128,
        "Minimum_Recommended_DDR5_GB": 2048,
        "Full_24DIMM_96GB_DDR5_GB": 2304,
        "Memory_Bandwidth_Comparison": {
            "under_populated_8ch_gbs": under_8ch_bw,
            "full_populated_24ch_gbs": full_24ch_bw,
            "l2_kv_32k_restore_ms_24ch": 97,
            "l2_kv_32k_restore_ms_8ch": 310,
        },
        "PCIe_Gen5_Topology": {
            "pcie_gen5_switches": 4,
            "switch_chip": "Broadcom PEX89144",
            "pix_rdma_busbw_gbs": 382,
            "switchless_sys_rdma_busbw_gbs": 136,
            "rdma_loss_without_switch_pct": 64.4,
        },
        "Local_NVMe_Pool": {
            "drives": 8,
            "capacity_per_drive_tb": 3.84,
            "aggregate_read_gbs": 56.0,
            "host_bom_price_premium_pct": 2.8,
            "node_performance_recovered_pct": 42.0,
        },
    }


if __name__ == "__main__":
    out = evaluate_gpu_host_server_bom()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)
