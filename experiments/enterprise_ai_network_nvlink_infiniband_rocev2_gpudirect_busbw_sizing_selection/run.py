from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: 400G RoCEv2 with PCIe ACS Enabled / Missing nvidia-peermem (CPU Bounce Anti-Pattern)",
            "nccl_allreduce_algbw_gbps": 6.0,
            "nccl_allreduce_busbw_gbps": 11.2,
            "line_rate_efficiency_pct": 22.4,
            "p99_alltoall_latency_us": 1420.0,
            "network_capex_per_gpu_usd": 2400,
        },
        {
            "tier": "Tier 2: Standard 400G RoCEv2 + Static 5-Tuple ECMP + Naive PFC Only",
            "nccl_allreduce_algbw_gbps": 16.8,
            "nccl_allreduce_busbw_gbps": 31.5,
            "line_rate_efficiency_pct": 63.0,
            "p99_alltoall_latency_us": 485.0,
            "network_capex_per_gpu_usd": 2400,
        },
        {
            "tier": "Tier 3: 400G AI Ethernet RoCEv2 (Rail-Optimized + GPUDirect + ECN/DCQCN + DLB/Packet Spray)",
            "nccl_allreduce_algbw_gbps": 25.4,
            "nccl_allreduce_busbw_gbps": 47.6,
            "line_rate_efficiency_pct": 95.2,
            "p99_alltoall_latency_us": 178.0,
            "network_capex_per_gpu_usd": 2650,
        },
        {
            "tier": "Tier 4: 400G/800G InfiniBand NDR/XDR (Native Credit Flow Control + Adaptive Routing + SHARPv3)",
            "nccl_allreduce_algbw_gbps": 38.6,
            "nccl_allreduce_busbw_gbps": 48.8,
            "line_rate_efficiency_pct": 97.6,
            "p99_alltoall_latency_us": 132.0,
            "network_capex_per_gpu_usd": 4150,
        },
    ]
    return {
        "experiment_id": "enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "interconnect_hierarchy": {
            "nvlink4_h200_bidirectional_gbps": 900.0,
            "nvlink5_b200_nvl72_bidirectional_gbps": 1800.0,
            "pcie_gen5_x16_bidirectional_gbps": 128.0,
            "rdma_400g_unidirectional_gbps": 50.0,
        },
        "tiers": tiers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_results()

    if args.check:
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing == payload, "results.json drift detected"
        print("OK: enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
