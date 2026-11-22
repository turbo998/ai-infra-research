from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_results() -> dict:
    cascade_tiers = [
        {
            "tier": "Tier 1: 100% Frontier Teacher Model Serving (All Traffic on 671B MoE / 405B Dense)",
            "domain_task_accuracy_pct": 95.4,
            "weighted_p99_tpot_ms": 18.5,
            "gpus_required_at_50_rps": 64,
            "monthly_serving_cost_usd": 115200.0,
        },
        {
            "tier": "Tier 2: Off-the-Shelf Un-Distilled 8B/14B Small Model Only",
            "domain_task_accuracy_pct": 74.2,
            "weighted_p99_tpot_ms": 5.2,
            "gpus_required_at_50_rps": 8,
            "monthly_serving_cost_usd": 14400.0,
        },
        {
            "tier": "Tier 3: Sequential Cascade (Try 8B First -> If Low LogProb Confidence, Re-Run Full Prompt on 671B)",
            "domain_task_accuracy_pct": 94.6,
            "weighted_p99_tpot_ms": 24.8,
            "gpus_required_at_50_rps": 28,
            "monthly_serving_cost_usd": 50400.0,
        },
        {
            "tier": "Tier 4: Domain-Distilled 32B/8B Student (82% Pre-Routed) + 671B Teacher (18% Hard Queries, Continuous Online Flywheel)",
            "domain_task_accuracy_pct": 95.1,
            "weighted_p99_tpot_ms": 7.4,
            "gpus_required_at_50_rps": 16,
            "monthly_serving_cost_usd": 28800.0,
        },
    ]
    return {
        "experiment_id": "enterprise_model_cascade_distillation_roi_teacher_student_routing_selection",
        "Provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "one_time_distillation_training_cost_usd": 1850.0,
        "distillation_breakeven_days": 0.64,
        "cascade_tiers": cascade_tiers,
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
        print("OK: enterprise_model_cascade_distillation_roi_teacher_student_routing_selection verified.")
        return

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")


if __name__ == "__main__":
    main()
