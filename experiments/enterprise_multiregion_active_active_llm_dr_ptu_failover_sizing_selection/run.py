from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    dr_architectures = [
        {
            "arch_id": "single_region_client_retry_storm",
            "name": "Anti-Pattern: Single-Region Endpoint + Naive Client Exponential Backoff Retries",
            "steady_capacity_allocation": "100% Region A (Zero Cross-Region Fallback)",
            "failover_rto_seconds": 45.0,
            "effective_annual_sla_pct": 99.45,
            "dr_idle_capacity_tax_pct": 0.0,
            "annual_platform_tco_usd": 240000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "active_passive_100_plus_100_standby",
            "name": "Legacy DR: Active-Passive 100% Primary + 100% Idle Standby PTU/GPU Pool",
            "steady_capacity_allocation": "100% Region A Active + 100% Region B Cold/Warm Idle",
            "failover_rto_seconds": 1.8,
            "effective_annual_sla_pct": 99.98,
            "dr_idle_capacity_tax_pct": 100.0,
            "annual_platform_tco_usd": 480000.0,
            "action": "Send to a human",
        },
        {
            "arch_id": "active_active_55_55_apim_hedging_paygo_spillover",
            "name": "Optimal: Active-Active 55%+55% Dual-Region + 800ms TTFT Circuit Breaker + PAYGO Spillover",
            "steady_capacity_allocation": "55% Region A + 55% Region B Active + Cross-Cloud PAYGO Burst",
            "failover_rto_seconds": 0.85,
            "effective_annual_sla_pct": 99.995,
            "dr_idle_capacity_tax_pct": 10.0,
            "annual_platform_tco_usd": 276000.0,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_multiregion_active_active_llm_dr_ptu_failover_sizing_selection",
        "title": "Enterprise Multi-Region Active-Active LLM Disaster Recovery (99.995% SLA, 0.85s RTO) & Zero-Waste Failover Sizer (#516)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "summary_metrics": {
            "rto_speedup_factor_vs_single_region": round(45.0 / 0.85, 2),
            "tco_saving_pct_vs_active_passive": round((1.0 - 276000.0 / 480000.0) * 100, 1),
            "effective_sla_pct": 99.995,
        },
        "dr_architectures": dr_architectures,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
