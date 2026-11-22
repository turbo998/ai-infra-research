from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Manual 20-Prompt 'Vibe Check' (Ad-Hoc Human Review)",
            "regression_catch_rate_pct": 28.0,
            "human_agreement_kappa": 0.45,
            "ci_gate_latency_minutes": 240.0,
            "cost_per_release_gate_usd": 800.0,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: Brute-Force 50,000-Prompt Cloud Judge (Uncalibrated 1-10 Scale)",
            "regression_catch_rate_pct": 84.0,
            "human_agreement_kappa": 0.51,
            "ci_gate_latency_minutes": 390.0,
            "cost_per_release_gate_usd": 1250.0,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: Public Academic Benchmarks Only (MMLU / GSM8K)",
            "regression_catch_rate_pct": 35.0,
            "human_agreement_kappa": 0.42,
            "ci_gate_latency_minutes": 45.0,
            "cost_per_release_gate_usd": 45.0,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: Stratified 200-Case Gate + Position-Swapped Binary 14B Judge + 1% Shadow Canary (Recommended)",
            "regression_catch_rate_pct": 98.4,
            "human_agreement_kappa": 0.89,
            "ci_gate_latency_minutes": 2.8,
            "cost_per_release_gate_usd": 3.4,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_llm_eval_cicd_judge_calibration_shadow_canary_selection",
        "title": "Milestone #505: Enterprise LLM Evaluation, CI/CD Regression Gate, Judge Calibration & Shadow Canary Selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "eval_pipeline_metrics": {
            "speedup_vs_50k_bruteforce": round(390.0 / 2.8, 1),
            "cost_reduction_pct_vs_50k": round((1.0 - 3.4 / 1250.0) * 100, 1),
            "kappa_improvement_pts": round(0.89 - 0.51, 2),
        },
        "tiers": tiers,
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = build_results()
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
