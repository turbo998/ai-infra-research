from __future__ import annotations

import json
from pathlib import Path


def evaluate_thinking_budget_tiers() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Unbounded Single max_tokens=4096 + Keep Historical <think>",
            "empty_answer_truncation_pct": 29.4,
            "task_completion_accuracy_pct": 65.8,
            "multi_turn_kv_memory_gb": 14.40,
            "malformed_json_tool_pct": 29.4,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 2: Strip Historical <think> Only (No Thinking Budget Forcing)",
            "empty_answer_truncation_pct": 29.4,
            "task_completion_accuracy_pct": 70.6,
            "multi_turn_kv_memory_gb": 4.50,
            "malformed_json_tool_pct": 29.4,
            "decision": "Send to a human",
        },
        {
            "tier": "Tier 3: Abrupt Mid-Sentence </think> Token Forcing at B_think=3072",
            "empty_answer_truncation_pct": 0.0,
            "task_completion_accuracy_pct": 84.2,
            "multi_turn_kv_memory_gb": 4.50,
            "malformed_json_tool_pct": 14.2,
            "decision": "Confirm",
        },
        {
            "tier": "Tier 4: Soft-Newline Transition Bridge + Reserved Answer Budget + <think> Strip",
            "empty_answer_truncation_pct": 0.0,
            "task_completion_accuracy_pct": 94.8,
            "multi_turn_kv_memory_gb": 4.50,
            "malformed_json_tool_pct": 0.4,
            "decision": "Act automatically",
        },
    ]

    budget_splits = [
        {"max_tokens": 2048, "unbounded_empty_pct": 54.0, "tier4_empty_pct": 0.0, "tier4_acc_pct": 88.5},
        {"max_tokens": 4096, "unbounded_empty_pct": 29.4, "tier4_empty_pct": 0.0, "tier4_acc_pct": 94.8},
        {"max_tokens": 8192, "unbounded_empty_pct": 11.8, "tier4_empty_pct": 0.0, "tier4_acc_pct": 96.6},
        {"max_tokens": 16384, "unbounded_empty_pct": 3.5, "tier4_empty_pct": 0.0, "tier4_acc_pct": 97.2},
    ]

    summary = {
        "benchmark": "llm_reasoning_thinking_budget_forcing_empty_content_truncation_stream_parser_selection",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)",
        "total_max_tokens": 4096,
        "reserved_thinking_tokens": 3072,
        "reserved_answer_tokens": 1024,
        "naive_empty_answer_pct": 29.4,
        "tier4_empty_answer_pct": 0.0,
        "naive_task_accuracy_pct": 65.8,
        "tier4_task_accuracy_pct": 94.8,
        "accuracy_gain_pt": round(94.8 - 65.8, 1),
        "kv_memory_reduction_x": round(14.40 / 4.50, 2),
        "kv_memory_saved_pct": round((1.0 - 4.50 / 14.40) * 100.0, 1),
        "tiers": tiers,
        "budget_splits": budget_splits,
    }
    return summary


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    results = evaluate_thinking_budget_tiers()
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
