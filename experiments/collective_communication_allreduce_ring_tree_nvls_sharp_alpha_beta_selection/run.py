from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "collective_communication_allreduce_ring_tree_nvls_sharp_alpha_beta_selection",
        "title": "AI Cluster Collective Communication (All-Reduce/AG/RS/A2A): Ring vs. Tree vs. NVLS/SHARPv3 Alpha-Beta Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "cluster_gpus": 1024,
            "nodes_8gpu": 128,
            "small_tp_payload_mb": 4.0,
            "large_dp_payload_gb": 4.0,
        },
        "tiers": [
            {
                "tier_id": "tier1_forced_flat_ring_allreduce_across_all_sizes_and_1024_gpus",
                "name": "Tier 1: Forced Flat Ring All-Reduce Across All Message Sizes (`2(N-1)` = 2,046 Hop Latency Wall!)",
                "small_4mb_allreduce_latency_us": 4120.0,
                "large_4gb_allreduce_algbw_gb_s": 21.8,
                "straggler_single_slow_link_amplification_x": 4.9,
                "monthly_1024gpu_comm_stall_tco_usd": 680000,
                "verdict": "Anti-Pattern: 2,046 serial hops destroy small/medium collective latency (4.12ms for 4MB!)",
            },
            {
                "tier_id": "tier2_standard_ring_plus_double_binary_tree_without_nvls_or_sharp",
                "name": "Tier 2: Adaptive Ring (Large >=256MB) + Double Binary Tree (Small/Med, `2*log2(N)` = 20 Hops)",
                "small_4mb_allreduce_latency_us": 185.0,
                "large_4gb_allreduce_algbw_gb_s": 23.4,
                "straggler_single_slow_link_amplification_x": 2.4,
                "monthly_1024gpu_comm_stall_tco_usd": 310000,
                "verdict": "Tree cuts 1,024-GPU small-payload latency 22.3x (185us), but large All-Reduce still sends 2x data over wire",
            },
            {
                "tier_id": "tier3_intra_node_nvls_nvswitch_plus_inter_node_tree_ring",
                "name": "Tier 3: Intra-Node NVLS (NVSwitch SHARP) + Inter-Node Hierarchical Tree/Ring (RoCEv2 / IB)",
                "small_4mb_allreduce_latency_us": 92.0,
                "large_4gb_allreduce_algbw_gb_s": 29.8,
                "straggler_single_slow_link_amplification_x": 1.45,
                "monthly_1024gpu_comm_stall_tco_usd": 215000,
                "verdict": "Great for RoCEv2 & IB clusters: NVSwitch hardware reduction boosts intra-node TP & hierarchical DP by +27%",
            },
            {
                "tier_id": "tier4_hierarchical_nvls_plus_inter_node_sharpv3_collnet_and_flight_recorder",
                "name": "Tier 4: Hierarchical NVLS (Scale-Up) + SHARPv3 In-Network Reduction (Scale-Out) + PyTorch FlightRecorder",
                "small_4mb_allreduce_latency_us": 54.0,
                "large_4gb_allreduce_algbw_gb_s": 38.6,
                "straggler_single_slow_link_amplification_x": 1.08,
                "monthly_1024gpu_comm_stall_tco_usd": 148000,
                "verdict": "Recommended: 54us small All-Reduce (-76.3x), 38.6 GB/s algbw (+77.1% via SHARPv3) & -78.2% stall TCO",
            },
        ],
        "headline_findings": {
            "small_4mb_latency_us": 54.0,
            "large_4gb_algbw_gb_s": 38.6,
            "algbw_gain_vs_tier1_pct": 77.1,
            "tco_savings_vs_tier1_pct": 78.2,
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
