from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    skus = 50000
    tiers = [
        {
            "tier": "Tier 1: Legacy ERP Moving-Average / ARIMA + Manual Spreadsheet Replenishment",
            "forecast_wmape_pct": 26.4,
            "hard_constraint_violation_pct": 4.5,
            "stockout_rate_pct": 7.4,
            "disruption_replan_hours": 72.0,
            "annual_working_capital_and_platform_cost_musd": 18.5,
        },
        {
            "tier": "Tier 2: Pure Text LLM Prompting Over CSV Tables (Direct Forecast & PO Generation)",
            "forecast_wmape_pct": 31.5,
            "hard_constraint_violation_pct": 38.0,
            "stockout_rate_pct": 8.8,
            "disruption_replan_hours": 1.5,
            "annual_working_capital_and_platform_cost_musd": 22.4,
        },
        {
            "tier": "Tier 3: Commercial Monolithic APS Suite (Blue Yonder / Kinaxis / SAP IBP Statistical Only)",
            "forecast_wmape_pct": 14.8,
            "hard_constraint_violation_pct": 0.0,
            "stockout_rate_pct": 3.1,
            "disruption_replan_hours": 18.0,
            "annual_working_capital_and_platform_cost_musd": 11.2,
        },
        {
            "tier": "Tier 4: Neuro-Symbolic Control Tower (LLM Disruption Sensor + Chronos/LightGBM + OR-Tools MILP Solver)",
            "forecast_wmape_pct": 10.6,
            "hard_constraint_violation_pct": 0.0,
            "stockout_rate_pct": 1.8,
            "disruption_replan_hours": 0.2,
            "annual_working_capital_and_platform_cost_musd": 6.4,
        },
    ]
    return {
        "experiment_id": "enterprise_supply_chain_control_tower_chronos_milp_agent_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "active_skus": skus,
            "warehouses": 20,
            "annual_gmv_musd": 500.0,
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
        print("OK: enterprise_supply_chain_control_tower_chronos_milp_agent_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
