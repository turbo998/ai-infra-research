from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    acceptance_tiers = [
        {
            "tier": "Tier 1: Zero Burn-In (Accept Nodes Once nvidia-smi & ibstat Show UP)",
            "doa_and_lemon_nodes_caught_before_prod_pct": 12.0,
            "first_month_production_job_abort_rate_pct": 38.5,
            "sustained_cluster_goodput_pct": 54.0,
            "silent_data_corruption_caught": False,
        },
        {
            "tier": "Tier 2: Single-Node 1-Hour Smoke Test Only (No All-Pairs RDMA Sweep or Thermal Soak)",
            "doa_and_lemon_nodes_caught_before_prod_pct": 48.0,
            "first_month_production_job_abort_rate_pct": 18.2,
            "sustained_cluster_goodput_pct": 74.5,
            "silent_data_corruption_caught": False,
        },
        {
            "tier": "Tier 3: 24-Hour DCGM Level-4 Burn-In + Full-Cluster nccl-tests AllReduce",
            "doa_and_lemon_nodes_caught_before_prod_pct": 88.5,
            "first_month_production_job_abort_rate_pct": 4.1,
            "sustained_cluster_goodput_pct": 91.2,
            "silent_data_corruption_caught": True,
        },
        {
            "tier": "Tier 4: 5-Stage Day-0 to Day-2 Acceptance Gate (72h Thermal/SDC Burn-In + Bisection Pairwise RDMA + GDS + AIPerf SLA + Chaos Drill)",
            "doa_and_lemon_nodes_caught_before_prod_pct": 99.6,
            "first_month_production_job_abort_rate_pct": 0.3,
            "sustained_cluster_goodput_pct": 98.4,
            "silent_data_corruption_caught": True,
        },
    ]
    return {
        "experiment_id": "enterprise_day0_day2_ai_cluster_acceptance_burnin_chaos_golive_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "observed_new_cluster_lemon_gpu_rate_pct": 3.2,
        "acceptance_tiers": acceptance_tiers,
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
        print("OK: enterprise_day0_day2_ai_cluster_acceptance_burnin_chaos_golive_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
