from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    dr_tiers = [
        {
            "tier": "Tier 1: Single-Region Deployment without Cross-Region Failover (Regional Outage / Quota Exhaustion Exposed)",
            "effective_annual_availability_pct": 99.40,
            "annual_unplanned_downtime_minutes": 3153.6,
            "failover_rto_seconds": 3600.0,
            "mid_stream_sse_drop_rate_on_failover_pct": 100.0,
            "standby_capacity_cost_overhead_pct": 0.0,
        },
        {
            "tier": "Tier 2: Active-Passive Cold Standby (On-Demand GPU Scale-From-Zero + 685GB Weight Pull on Outage)",
            "effective_annual_availability_pct": 99.82,
            "annual_unplanned_downtime_minutes": 946.1,
            "failover_rto_seconds": 780.0,
            "mid_stream_sse_drop_rate_on_failover_pct": 100.0,
            "standby_capacity_cost_overhead_pct": 12.0,
        },
        {
            "tier": "Tier 3: Dual-Region 100%+100% Idle Hot Standby (Doubled Idle GPU / PTU Reservation)",
            "effective_annual_availability_pct": 99.98,
            "annual_unplanned_downtime_minutes": 105.1,
            "failover_rto_seconds": 8.5,
            "mid_stream_sse_drop_rate_on_failover_pct": 42.0,
            "standby_capacity_cost_overhead_pct": 100.0,
        },
        {
            "tier": "Tier 4: Dual-Region 60/60 Active-Active Base + Cross-Region PAYGO Spillover + Token-Offset SSE Resume",
            "effective_annual_availability_pct": 99.995,
            "annual_unplanned_downtime_minutes": 26.3,
            "failover_rto_seconds": 1.6,
            "mid_stream_sse_drop_rate_on_failover_pct": 0.0,
            "standby_capacity_cost_overhead_pct": 18.5,
        },
    ]
    return {
        "experiment_id": "enterprise_multi_region_active_active_ai_inference_disaster_recovery_ha_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "target_sla_nines": 99.99,
        "dr_tiers": dr_tiers,
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
        print("OK: enterprise_multi_region_active_active_ai_inference_disaster_recovery_ha_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
