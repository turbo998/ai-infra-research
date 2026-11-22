from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Server-Based ToR Fat-Tree + 2:1 Oversubscribed Spine + Static 5-Tuple ECMP",
            "spine_traffic_traversal_pct": 87.5,
            "effective_busbw_gbps": 24.2,
            "p99_alltoall_tail_us": 680.0,
            "switch_optics_capex_512gpu_usd": 1480000,
        },
        {
            "tier": "Tier 2: Non-Blocking 1:1 Fat-Tree + Static ECMP (No Rail Alignment / No PXN)",
            "spine_traffic_traversal_pct": 75.0,
            "effective_busbw_gbps": 32.8,
            "p99_alltoall_tail_us": 440.0,
            "switch_optics_capex_512gpu_usd": 1960000,
        },
        {
            "tier": "Tier 3: Rail-Optimized 8-Rail Leaf-Spine + NCCL PXN + Flowlet DLB",
            "spine_traffic_traversal_pct": 12.5,
            "effective_busbw_gbps": 46.8,
            "p99_alltoall_tail_us": 195.0,
            "switch_optics_capex_512gpu_usd": 1320000,
        },
        {
            "tier": "Tier 4: Rail-Optimized + NCCL PXN + Packet Spraying & Out-of-Order Direct Data Placement (Spectrum-X / UEC)",
            "spine_traffic_traversal_pct": 8.0,
            "effective_busbw_gbps": 48.2,
            "p99_alltoall_tail_us": 142.0,
            "switch_optics_capex_512gpu_usd": 1420000,
        },
    ]
    return {
        "experiment_id": "enterprise_rocev2_rail_optimized_pxn_dlb_packet_spraying_topology_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "topology_baseline": {
            "cluster_gpus": 512,
            "rails_per_node": 8,
            "per_nic_line_rate_gbps": 50.0,
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
        print("OK: enterprise_rocev2_rail_optimized_pxn_dlb_packet_spraying_topology_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
