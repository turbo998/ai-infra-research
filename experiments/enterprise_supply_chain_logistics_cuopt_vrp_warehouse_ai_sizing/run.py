"""Reproducible model for #611 Enterprise Supply Chain & Logistics AI (LLM Constraint Compiler + cuOpt VRP)."""
from __future__ import annotations
import json
import math
from pathlib import Path


def evaluate_supply_chain_logistics(trucks: int = 5000, daily_waybills: int = 20_000) -> dict:
    llm_parse_sec = 0.4
    cuopt_solve_sec = 1.8
    total_reopt_sec = round(llm_parse_sec + cuopt_solve_sec, 1)
    cpu_mip_baseline_sec = 2700.0
    speedup_ratio = round(cpu_mip_baseline_sec / total_reopt_sec, 1)

    l40s_gpus = max(2, math.ceil(daily_waybills / 5000.0))
    h100_gpus = max(2, math.ceil(trucks / 1250.0))
    monthly_gpu_cost_usd = l40s_gpus * 950 + h100_gpus * 2300

    annual_fuel_savings_million_usd = round((trucks * 2240) / 1_000_000, 2)
    payback_hours_per_month = round((monthly_gpu_cost_usd / ((trucks * 2240) / 12.0)) * 720.0, 1)

    return {
        "topic_id": 611,
        "trucks": trucks,
        "daily_waybills": daily_waybills,
        "total_reopt_sec": total_reopt_sec,
        "cpu_mip_baseline_sec": cpu_mip_baseline_sec,
        "speedup_ratio": speedup_ratio,
        "l40s_gpus": l40s_gpus,
        "h100_gpus": h100_gpus,
        "monthly_gpu_cost_usd": monthly_gpu_cost_usd,
        "annual_fuel_savings_million_usd": annual_fuel_savings_million_usd,
        "payback_hours_per_month": payback_hours_per_month,
        "time_window_violation_pct": 0.0,
    }


def main() -> None:
    res = evaluate_supply_chain_logistics()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
