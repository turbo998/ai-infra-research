from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_Payload() -> dict:
    return {
        "round": 407,
        "slug": "enterprise_supply_chain_control_tower_agent_selection",
        "title": "Enterprise Supply Chain Control Tower, Demand Forecasting & Disruption Mitigation Agent Architecture Selection",
        "Provenance": "synthetic_analytical_model_calibrated_to_published_supply_chain_chronos_lightgbm_ortools_milp_and_neuro_symbolic_control_tower_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "active_skus": 250000,
            "distribution_warehouses": 40,
            "monthly_purchase_orders": 15000,
            "enterprise_sla": ">= 90% demand forecast accuracy (1 - WAPE), 0.0% MOQ/capacity constraint violations, <= 0.5h disruption mitigation turnaround",
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_erp_moving_average_manual_excel",
                "name": "Tier 1: Legacy ERP Static Moving-Average + Manual Excel Expediting",
                "forecast_accuracy_1_minus_wape_pct": 68.5,
                "moq_capacity_violation_rate_pct": 4.2,
                "disruption_recovery_hours": 72.0,
                "monthly_total_cost_usd": 145000,
                "action": "Send to a human",
                "status": "fail",
            },
            {
                "tier_id": "tier2_pure_llm_text_time_series_forecaster",
                "name": "Tier 2: Pure Generative LLM Prompted to Predict Time-Series Numbers & Order Quantities",
                "forecast_accuracy_1_minus_wape_pct": 62.0,
                "moq_capacity_violation_rate_pct": 18.5,
                "disruption_recovery_hours": 1.5,
                "monthly_total_cost_usd": 210000,
                "action": "Send to a human",
                "status": "fail",
            },
            {
                "tier_id": "tier3_chronos_lightgbm_plus_unconstrained_llm_po",
                "name": "Tier 3: Time-Series Model (Chronos/LightGBM) + Unconstrained LLM Order Recommendation",
                "forecast_accuracy_1_minus_wape_pct": 85.4,
                "moq_capacity_violation_rate_pct": 6.8,
                "disruption_recovery_hours": 2.0,
                "monthly_total_cost_usd": 64000,
                "action": "Confirm",
                "status": "warn",
            },
            {
                "tier_id": "tier4_llm_shock_covariates_chronos_ortools_milp_control_tower",
                "name": "Tier 4: LLM Exogenous Disruption Covariates + Chronos/LightGBM + OR-Tools MILP Solver + ERP PO HITL Agent",
                "forecast_accuracy_1_minus_wape_pct": 91.2,
                "moq_capacity_violation_rate_pct": 0.0,
                "disruption_recovery_hours": 0.25,
                "monthly_total_cost_usd": 22000,
                "action": "Act automatically",
                "status": "pass",
            },
        ],
        "robustness_check": [
            {
                "metric": "Demand Forecast Accuracy 1-WAPE (%)",
                "baseline_tier1": 68.5,
                "optimized_tier4": 91.2,
                "delta": "+22.7 pts (+29.2 pts vs Tier 2)",
            },
            {
                "metric": "MOQ / Capacity Constraint Violation (%)",
                "baseline_tier2": 18.5,
                "optimized_tier4": 0.0,
                "delta": "-18.5 pts (0.0% via MILP solver)",
            },
            {
                "metric": "Disruption Mitigation Lead Time (Hours)",
                "baseline_tier1": 72.0,
                "optimized_tier4": 0.25,
                "delta": "288x faster (15-min PO reallocation)",
            },
            {
                "metric": "Monthly Expedite & Ops Cost (USD)",
                "baseline_tier1": 145000,
                "optimized_tier4": 22000,
                "delta": "-84.8% (-$123,000/mo)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    payload = build_Payload()

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SystemExit("results.json out of sync")
        print("OK: enterprise_supply_chain_control_tower_agent_selection verified.")
        return 0

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
