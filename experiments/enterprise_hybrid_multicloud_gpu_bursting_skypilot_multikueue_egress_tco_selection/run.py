from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    bursting_models = [
        {
            "model": "Tier 1: 100% On-Premises Peak Over-Provisioning (Sized for 3x Holiday/Morning Peak, Idle 78% of Hours)",
            "on_prem_base_gpus": 128,
            "cloud_burst_egress_bill_monthly_usd": 0.0,
            "cross_wan_latency_penalty_ms": 0.0,
            "annual_total_tco_usd": 2764800.0,
        },
        {
            "model": "Tier 2: Naive Cross-WAN PD Disaggregation or Remote Storage Pull (On-Prem Prefill -> Cloud Decode over WAN)",
            "on_prem_base_gpus": 64,
            "cloud_burst_egress_bill_monthly_usd": 48500.0,
            "cross_wan_latency_penalty_ms": 940.0,
            "annual_total_tco_usd": 2180000.0,
        },
        {
            "model": "Tier 3: Single-Cloud Bursting with On-Demand Cold Weight Pull on Every Scale-Out",
            "on_prem_base_gpus": 48,
            "cloud_burst_egress_bill_monthly_usd": 14200.0,
            "cross_wan_latency_penalty_ms": 38.0,
            "annual_total_tco_usd": 1490000.0,
        },
        {
            "model": "Tier 4: On-Prem P80 Base (48 GPUs) + Multi-Cloud SkyPilot/MultiKueue Whole-Request Burst + Pre-Staged R2/VPC Weights",
            "on_prem_base_gpus": 48,
            "cloud_burst_egress_bill_monthly_usd": 420.0,
            "cross_wan_latency_penalty_ms": 14.0,
            "annual_total_tco_usd": 1185000.0,
        },
    ]
    return {
        "experiment_id": "enterprise_hybrid_multicloud_gpu_bursting_skypilot_multikueue_egress_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "peak_gpu_demand": 128,
        "p80_base_gpu_demand": 48,
        "bursting_models": bursting_models,
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
        print("OK: enterprise_hybrid_multicloud_gpu_bursting_skypilot_multikueue_egress_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
