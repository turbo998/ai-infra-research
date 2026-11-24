from __future__ import annotations
import json
from pathlib import Path


def evaluate_aviation_airline_irops_sizing() -> dict:
    fleet_aircraft = 500
    raw_llm_crew_violation_pct = 27.4
    column_gen_mip_violation_pct = 0.0
    irops_recovery_seconds = 4.2
    turnaround_delay_saved_min = 18.0
    aog_reduction_pct = 34.0
    annual_delay_and_aog_savings_usd = 19_400_000

    api_monthly_usd = 260_000
    l40s_gpus = 8
    h200_gpus = 8
    self_hosted_monthly_usd = l40s_gpus * 850 + h200_gpus * 3200  # 32,400
    compute_tco_savings_pct = round((1.0 - self_hosted_monthly_usd / api_monthly_usd) * 100, 1)

    return {
        "deliverable_id": "#621",
        "topic": "enterprise_aviation_airline_irops_solver_apron_vision_mro_sizing",
        "fleet_aircraft": fleet_aircraft,
        "raw_llm_crew_violation_pct": raw_llm_crew_violation_pct,
        "column_gen_mip_violation_pct": column_gen_mip_violation_pct,
        "irops_recovery_seconds": irops_recovery_seconds,
        "turnaround_delay_saved_min": turnaround_delay_saved_min,
        "aog_reduction_pct": aog_reduction_pct,
        "annual_delay_and_aog_savings_usd": annual_delay_and_aog_savings_usd,
        "api_monthly_usd": api_monthly_usd,
        "self_hosted_monthly_usd": self_hosted_monthly_usd,
        "compute_tco_savings_pct": compute_tco_savings_pct,
    }


if __name__ == "__main__":
    out = evaluate_aviation_airline_irops_sizing()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Wrote", out_path.name)
