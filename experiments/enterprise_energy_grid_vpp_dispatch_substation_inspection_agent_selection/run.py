from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_energy_grid_vpp_dispatch_substation_inspection_agent_selection_v1",
        "title": "Enterprise Energy Grid Virtual Power Plant (VPP) Dispatch, Radiometric Substation Inspection VLM & Five-Preventions Switching-Order Agent Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "aggregated_vpp_capacity_mw": 2400,
            "substations_110kv_220kv": 320,
            "monthly_drone_thermal_rgb_images": 450000,
            "monthly_switching_and_work_orders": 3800,
            "sla_target": "VPP forecast accuracy >= 93%, Five-Preventions switching safety violation rate == 0.0%, thermal hotspot recall >= 96%",
        },
        "tiers": [
            {
                "tier_id": "tier1_manual_trader_bidding_and_manual_substation_inspection",
                "name": "Tier 1: Manual Spreadsheet VPP Bidding + Manual Drone Photo Review & Switching Orders",
                "vpp_renewable_forecast_accuracy_pct": 76.5,
                "five_preventions_interlock_violation_rate_pct": 0.4,
                "thermal_hotspot_defect_recall_pct": 72.0,
                "monthly_imbalance_penalty_and_ops_cost_usd": 295000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_prompt_only_llm_power_bidding_and_unverified_switching_order",
                "name": "Tier 2: Prompt-Only Text LLM Predicting MW Curves & Drafting Unverified Switching Orders",
                "vpp_renewable_forecast_accuracy_pct": 68.0,
                "five_preventions_interlock_violation_rate_pct": 14.5,
                "thermal_hotspot_defect_recall_pct": 64.5,
                "monthly_imbalance_penalty_and_ops_cost_usd": 420000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_nwp_timeseries_model_plus_text_only_grid_code_rag",
                "name": "Tier 3: NWP + Time-Series Forecaster + Text-Only Grid Code RAG",
                "vpp_renewable_forecast_accuracy_pct": 87.2,
                "five_preventions_interlock_violation_rate_pct": 4.8,
                "thermal_hotspot_defect_recall_pct": 81.0,
                "monthly_imbalance_penalty_and_ops_cost_usd": 128000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_weather_llm_stochastic_milp_vpp_plus_radiometric_vlm_plus_five_prevention_agent",
                "name": "Tier 4: NWP/Timeseries + Stochastic MILP VPP Solver + Radiometric ΔT VLM + Five-Preventions Topology Interlock Agent",
                "vpp_renewable_forecast_accuracy_pct": 94.4,
                "five_preventions_interlock_violation_rate_pct": 0.0,
                "thermal_hotspot_defect_recall_pct": 97.2,
                "monthly_imbalance_penalty_and_ops_cost_usd": 41000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "Cold-Front Cloud-Cover Ramp Event (-65% Solar Drop in 30m) + 220kV Double-Busbar Transfer Switching Order Verification",
            "tier2_switching_interlock_violation_pct": 22.0,
            "tier4_topology_interlock_violation_pct": 0.0,
            "tier1_vpp_forecast_accuracy_pct": 76.5,
            "tier4_vpp_forecast_accuracy_pct": 94.4,
            "forecast_gain_pts": 17.9,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    payload = build_payload()
    out_path = Path(__file__).resolve().parent / "results.json"
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            return 1
        return 0 if out_path.read_text(encoding="utf-8") == rendered else 1

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
