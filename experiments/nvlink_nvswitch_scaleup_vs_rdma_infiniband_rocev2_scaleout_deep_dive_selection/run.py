from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "nvlink_nvswitch_scaleup_vs_rdma_infiniband_rocev2_scaleout_deep_dive_selection",
        "title": "NVLink/NVSwitch (Scale-Up) vs. InfiniBand & RoCEv2 RDMA (Scale-Out): GPUDirect Path, Lossless Tuning & busbw Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "cluster_gpus": 512,
            "nodes_8gpu_hgx": 64,
            "nic_speed_gbps": 400,
            "theoretical_single_rail_400g_gb_s": 50.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_rocev2_without_peermem_or_ecn_static_ecmp",
                "name": "Tier 1: Un-tuned RoCEv2 (No nvidia-peermem GPUDirect, Static ECMP & Pure PFC Pause Storms)",
                "nccl_allreduce_busbw_gb_s": 11.8,
                "p99_collective_tail_latency_ms": 94.0,
                "monthly_pfc_deadlock_and_straggler_incidents": 14,
                "monthly_512gpu_network_and_wasted_compute_tco_usd": 1150000,
                "verdict": "Anti-Pattern: CPU RAM bounce + ECMP hash collisions drop 400G busbw to 11.8 GB/s (23.6% eff.)",
            },
            {
                "tier2_tuned_rocev2_rail_optimized_dcqcn_dlb_gpudirect": "tier2",
                "tier_id": "tier2_tuned_rocev2_rail_optimized_dcqcn_dlb_gpudirect",
                "name": "Tier 2: Tuned AI Ethernet RoCEv2 (GPUDirect RDMA + Rail-Optimized + ECN/DCQCN + DLB/Spray + Watchdog)",
                "nccl_allreduce_busbw_gb_s": 45.4,
                "p99_collective_tail_latency_ms": 8.2,
                "monthly_pfc_deadlock_and_straggler_incidents": 0,
                "monthly_512gpu_network_and_wasted_compute_tco_usd": 410000,
                "verdict": "Best TCO for <=512-1024 GPU training, PD serving & cloud-native pools (90.8% line-rate busbw)",
            },
            {
                "tier_id": "tier3_infiniband_ndr400_rail_fat_tree_adaptive_routing_sharp",
                "name": "Tier 3: InfiniBand NDR 400G (Credit-Based Lossless + Adaptive Routing + SHARPv3 In-Network Reduce)",
                "nccl_allreduce_busbw_gb_s": 48.2,
                "p99_collective_tail_latency_ms": 5.4,
                "monthly_pfc_deadlock_and_straggler_incidents": 0,
                "monthly_512gpu_network_and_wasted_compute_tco_usd": 465000,
                "verdict": "Best latency & zero-tuning lossless fabric for >=1,000-GPU synchronous pre-training & heavy MoE",
            },
            {
                "tier4_hierarchical_nvlink_nvls_scaleup_plus_tuned_rdma_scaleout": "tier4",
                "tier_id": "tier4_hierarchical_nvlink_nvls_scaleup_plus_tuned_rdma_scaleout",
                "name": "Tier 4: Hierarchical Scale-Up (NVLink/NVLS 900-1800 GB/s for TP) × Scale-Out (GPUDirect IB/RoCEv2 + PXN)",
                "nccl_allreduce_busbw_gb_s": 47.6,
                "p99_collective_tail_latency_ms": 4.9,
                "monthly_pfc_deadlock_and_straggler_incidents": 0,
                "monthly_512gpu_network_and_wasted_compute_tco_usd": 365000,
                "verdict": "Recommended: 47.6 GB/s scale-out busbw (95.2% eff.), 0 deadlocks & -68.3% TCO vs. Tier 1",
            },
        ],
        "headline_findings": {
            "scaleout_busbw_gb_s": 47.6,
            "line_rate_efficiency_pct": 95.2,
            "p99_collective_latency_ms": 4.9,
            "tco_savings_vs_tier1_pct": 68.3,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
