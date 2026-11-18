from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_payload() -> dict[str, Any]:
    return {
        "benchmark_id": "enterprise_smart_building_datacenter_hvac_mpc_energy_agent_selection_v1",
        "title": "Enterprise Smart Building & Data Center BMS HVAC Copilot: Brick Schema Ontology, Physics-Informed MPC & BACnet Safety Clamp Evaluator",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_baseline": {
            "campus_and_datacenter_buildings": 45,
            "centrifugal_chillers_and_cooling_towers": 180,
            "bacnet_modbus_telemetry_points": 350000,
            "sla_target": "HVAC energy & TOU demand savings >= 24%, ASHRAE thermal / chiller-surge violation rate == 0.0%, BMS point onboarding <= 6.0 hours/building",
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_bms_fixed_schedules_and_reactive_pid",
                "name": "Tier 1: Legacy BMS Fixed Schedules + Reactive Local PID Loops + Manual Point Tagging",
                "hvac_energy_and_peak_demand_savings_pct": 4.5,
                "ashrae_thermal_or_chiller_surge_violation_rate_pct": 1.5,
                "bms_point_onboarding_hours_per_building": 160.0,
                "monthly_hvac_electricity_and_ops_cost_usd": 420000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier2_prompt_only_llm_direct_bacnet_setpoint_control",
                "name": "Tier 2: Prompt-Only LLM Directly Writing BACnet Chiller & AHU Setpoints",
                "hvac_energy_and_peak_demand_savings_pct": 12.0,
                "ashrae_thermal_or_chiller_surge_violation_rate_pct": 13.8,
                "bms_point_onboarding_hours_per_building": 18.0,
                "monthly_hvac_electricity_and_ops_cost_usd": 510000,
                "action_decision": "Send to a human",
            },
            {
                "tier_id": "tier3_brick_schema_rag_plus_fdd_ticket_copilot",
                "name": "Tier 3: Brick Schema Tagging + Read-Only Fault Detection & Diagnostics (FDD) Copilot",
                "hvac_energy_and_peak_demand_savings_pct": 11.8,
                "ashrae_thermal_or_chiller_surge_violation_rate_pct": 0.8,
                "bms_point_onboarding_hours_per_building": 12.0,
                "monthly_hvac_electricity_and_ops_cost_usd": 215000,
                "action_decision": "Confirm",
            },
            {
                "tier_id": "tier4_llm_brick_ontology_mapper_plus_physics_mpc_solver_plus_bacnet_safety_clamp_agent",
                "name": "Tier 4: LLM Brick Schema Auto-Mapper + Physics-Informed MPC Chiller/TOU Solver + Local PLC BACnet Safety Clamp Agent",
                "hvac_energy_and_peak_demand_savings_pct": 26.4,
                "ashrae_thermal_or_chiller_surge_violation_rate_pct": 0.0,
                "bms_point_onboarding_hours_per_building": 4.5,
                "monthly_hvac_electricity_and_ops_cost_usd": 64000,
                "action_decision": "Act automatically",
            },
        ],
        "perturbation_check": {
            "scenario": "38°C Wet-Bulb Heatwave + 4x Peak TOU Electricity Price Window + Chiller Minimum-Flow Surge Boundary Test",
            "tier2_surge_or_thermal_excursion_pct": 24.5,
            "tier4_mpc_plc_clamp_excursion_pct": 0.0,
            "tier1_energy_savings_pct": 4.5,
            "tier4_energy_savings_pct": 26.4,
            "savings_gain_pts": 21.9,
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
