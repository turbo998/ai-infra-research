"""Reproducible model for #590 Milestone Cloud & AI SA 15-Scenario Enterprise AI Encyclopedia."""
from __future__ import annotations
import json
from pathlib import Path


def evaluate_milestone_590(scale_factor: float = 1.0) -> dict:
    pool_l40s = round(18 * scale_factor)
    pool_h100 = round(12 * scale_factor)
    pool_h200 = round(16 * scale_factor)
    unified_total_gpus = pool_l40s + pool_h100 + pool_h200
    siloed_total_gpus = round(248 * scale_factor)

    unified_annual_million_usd = round(2.64 * scale_factor, 2)
    siloed_annual_million_usd = round(14.20 * scale_factor, 2)
    annual_savings_million_usd = round(siloed_annual_million_usd - unified_annual_million_usd, 2)

    return {
        "topic_id": 590,
        "deliverables_synthesized": 28,
        "range": "#563-#590",
        "scenarios_covered": 15,
        "pool_l40s": pool_l40s,
        "pool_h100": pool_h100,
        "pool_h200": pool_h200,
        "unified_total_gpus": unified_total_gpus,
        "siloed_total_gpus": siloed_total_gpus,
        "gpu_reduction_pct": round((1.0 - unified_total_gpus / siloed_total_gpus) * 100, 1),
        "unified_annual_million_usd": unified_annual_million_usd,
        "siloed_annual_million_usd": siloed_annual_million_usd,
        "annual_savings_million_usd": annual_savings_million_usd,
        "tco_reduction_pct": round((1.0 - unified_annual_million_usd / siloed_annual_million_usd) * 100, 1),
    }


def main() -> None:
    res = evaluate_milestone_590()
    out_path = Path(__file__).with_name("results.json")
    out_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out_path.name)


if __name__ == "__main__":
    main()
