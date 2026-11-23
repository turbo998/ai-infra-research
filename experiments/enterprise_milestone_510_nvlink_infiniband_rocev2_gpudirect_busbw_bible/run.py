from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    fabrics = [
        {
            "fabric_id": "rocev2_cpu_bounce_legacy_ecmp",
            "name": "Anti-Pattern: 400G RoCEv2 without GPUDirect RDMA (CPU Bounce + Static 5-Tuple ECMP)",
            "scale_tier": "Scale-Out (Misconfigured)",
            "unidirectional_bw_gbps": 400.0,
            "nccl_allreduce_algbw_gbps": 5.8,
            "nccl_allreduce_busbw_gbps": 11.4,
            "p2p_latency_us": 18.5,
            "wire_efficiency_pct": 22.8,
            "relative_network_capex_pct": 67.5,
            "action": "Send to a human",
        },
        {
            "fabric_id": "rocev2_static_ecmp_no_dlb",
            "name": "Baseline: 400G RoCEv2 + GPUDirect RDMA (Static ECMP, Uncalibrated ECN/PFC)",
            "scale_tier": "Scale-Out Ethernet",
            "unidirectional_bw_gbps": 400.0,
            "nccl_allreduce_algbw_gbps": 10.9,
            "nccl_allreduce_busbw_gbps": 21.4,
            "p2p_latency_us": 6.8,
            "wire_efficiency_pct": 42.8,
            "relative_network_capex_pct": 67.5,
            "action": "Send to a human",
        },
        {
            "fabric_id": "ai_ethernet_rocev2_rail_dlb_dcqcn",
            "name": "Optimal <=512 GPU: 400G AI Ethernet RoCEv2 (GPUDirect + Rail-Optimized + ECN/DCQCN + DLB/Spray)",
            "scale_tier": "Scale-Out AI Ethernet",
            "unidirectional_bw_gbps": 400.0,
            "nccl_allreduce_algbw_gbps": 23.6,
            "nccl_allreduce_busbw_gbps": 46.4,
            "p2p_latency_us": 2.6,
            "wire_efficiency_pct": 92.8,
            "relative_network_capex_pct": 67.5,
            "action": "Act automatically",
        },
        {
            "fabric_id": "infiniband_ndr_400g_sharp",
            "name": "Optimal >=1024 GPU: 400G InfiniBand NDR (GPUDirect + Credit Flow + Adaptive Routing + SHARP)",
            "scale_tier": "Scale-Out InfiniBand",
            "unidirectional_bw_gbps": 400.0,
            "nccl_allreduce_algbw_gbps": 34.8,
            "nccl_allreduce_busbw_gbps": 48.2,
            "p2p_latency_us": 1.5,
            "wire_efficiency_pct": 96.4,
            "relative_network_capex_pct": 100.0,
            "action": "Act automatically",
        },
        {
            "fabric_id": "nvlink4_nvswitch_intra_node",
            "name": "Scale-Up Reference: Intra-Node NVLink 4.0 + NVSwitch (H100/H200 8-GPU HGX Domain)",
            "scale_tier": "Scale-Up NVLink",
            "unidirectional_bw_gbps": 3600.0,
            "nccl_allreduce_algbw_gbps": 212.0,
            "nccl_allreduce_busbw_gbps": 371.0,
            "p2p_latency_us": 0.8,
            "wire_efficiency_pct": 82.4,
            "relative_network_capex_pct": 0.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_milestone_510_nvlink_infiniband_rocev2_gpudirect_busbw_bible",
        "title": "Milestone #510: Enterprise AI Fabric Bible — Scale-Up NVLink vs. Scale-Out InfiniBand/RoCEv2, GPUDirect RDMA, PFC/ECN/DCQCN & algbw/busbw Sizer",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "gpudirect_rdma_busbw_speedup_vs_cpu_bounce": round(48.2 / 11.4, 2),
            "tuned_rocev2_vs_ib_ndr_busbw_parity_pct": round((46.4 / 48.2) * 100, 1),
            "ai_ethernet_capex_saving_vs_ib_pct": 32.5,
        },
        "fabrics": fabrics,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
