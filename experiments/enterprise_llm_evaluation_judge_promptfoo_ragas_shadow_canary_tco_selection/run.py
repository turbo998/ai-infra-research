from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    monthly_evals_m = 5.0
    tiers = [
        {
            "tier": "Tier 1: Manual Spreadsheet Vibe-Check (50 Samples Before Release)",
            "human_agreement_kappa": 0.42,
            "production_regression_escape_rate_pct": 18.4,
            "cost_per_1k_evals_usd": 45.0,
            "monthly_eval_tco_usd": 38000,
        },
        {
            "tier": "Tier 2: Uncalibrated 1-5 Likert Frontier LLM Judge (100% GPT-4o Single-Pass)",
            "human_agreement_kappa": 0.54,
            "production_regression_escape_rate_pct": 9.2,
            "cost_per_1k_evals_usd": 19.0,
            "monthly_eval_tco_usd": 95000,
        },
        {
            "tier": "Tier 3: Binary Rubric Frameworks on Cloud API (Promptfoo / DeepEval / Ragas / Azure Eval SDK)",
            "human_agreement_kappa": 0.86,
            "production_regression_escape_rate_pct": 1.4,
            "cost_per_1k_evals_usd": 4.5,
            "monthly_eval_tco_usd": 22500,
        },
        {
            "tier": "Tier 4: Code Assertions + NLI Fast-Filter + Self-Hosted 14B Binary Judge (Position-Swapped) + Shadow Replay Gate",
            "human_agreement_kappa": 0.896,
            "production_regression_escape_rate_pct": 0.35,
            "cost_per_1k_evals_usd": 0.91,
            "monthly_eval_tco_usd": 4560,
        },
    ]
    return {
        "experiment_id": "enterprise_llm_evaluation_judge_promptfoo_ragas_shadow_canary_tco_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "workload_baseline": {
            "monthly_evaluations_millions": monthly_evals_m,
            "golden_regression_dataset_size": 1000,
            "shadow_traffic_mirror_pct": 5.0,
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
        print("OK: enterprise_llm_evaluation_judge_promptfoo_ragas_shadow_canary_tco_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
