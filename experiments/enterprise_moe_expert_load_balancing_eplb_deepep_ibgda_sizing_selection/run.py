from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Static Expert Placement + Blocking NCCL All-to-All (Hotspot Straggler Stalls)",
            "max_to_avg_expert_skew_ratio": 3.45,
            "alltoall_exposed_latency_us": 620.0,
            "p99_tpot_ms": 29.8,
            "per_gpu_goodput_tok_s": 860,
        },
        {
            "tier": "Tier 2: Static Expert Placement + DeepEP Normal/IBGDA Async Kernels (No EPLB Replication)",
            "max_to_avg_expert_skew_ratio": 3.45,
            "alltoall_exposed_latency_us": 210.0,
            "p99_tpot_ms": 19.4,
            "per_gpu_goodput_tok_s": 1480,
        },
        {
            "tier": "Tier 3: Hierarchical EPLB (Node-First Packing + 32 Redundant Hot Experts) + Blocking All-to-All",
            "max_to_avg_expert_skew_ratio": 1.18,
            "alltoall_exposed_latency_us": 390.0,
            "p99_tpot_ms": 17.6,
            "per_gpu_goodput_tok_s": 1720,
        },
        {
            "tier": "Tier 4: Hierarchical EPLB (256+32 Experts) + DeepEP Dual-Mode (Normal Prefill / IBGDA Decode) + Two-Microbatch Overlap",
            "max_to_avg_expert_skew_ratio": 1.12,
            "alltoall_exposed_latency_us": 38.0,
            "p99_tpot_ms": 11.6,
            "per_gpu_goodput_tok_s": 2460,
        },
    ]
    return {
        "experiment_id": "enterprise_moe_expert_load_balancing_eplb_deepep_ibgda_sizing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "eplb_memory_overhead_fp8": {
            "routed_experts_total": 256,
            "redundant_expert_slots": 32,
            "extra_hbm_per_gpu_gb": 2.4,
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
        print("OK: enterprise_moe_expert_load_balancing_eplb_deepep_ibgda_sizing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
