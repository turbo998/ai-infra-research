from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "reasoning_model_thinking_budget_difficulty_router_kv_storm_sizing_selection",
        "title": "Reasoning Model (DeepSeek-R1 / o3 / QwQ) Thinking-Token Budget Governance, 3-Tier Difficulty Routing & Little's Law KV Storm Sizing Guide",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "arrival_rate_rps": 40.0,
            "query_difficulty_distribution": "65% Simple / 25% Medium Analytical / 10% Hard Math/Code/Architecture",
        },
        "tiers": [
            {
                "tier": "Tier 1: Unbounded Full Reasoning on All Queries + Retaining <think> Blocks in Multi-Turn History",
                "avg_thinking_tokens": 3450,
                "peak_active_kv_sessions": 2450,
                "task_accuracy_pct": 94.2,
                "cost_per_task_usd": 0.0180,
                "recommended_scenario": "Severe anti-pattern causing 8.5x KV cache explosion and 72s latency on trivial prompts",
            },
            {
                "tier": "Tier 2: Uniform Hard Truncation (max_tokens=1024 Across All Queries Without </think> Grace Period)",
                "avg_thinking_tokens": 820,
                "peak_active_kv_sessions": 740,
                "task_accuracy_pct": 76.5,
                "cost_per_task_usd": 0.0065,
                "recommended_scenario": "Unacceptable accuracy collapse (-17.7 pts) on complex coding and reasoning tasks",
            },
            {
                "tier": "Tier 3: 3-Tier Difficulty Router (65% Fast / 25% 1K-Think / 10% 8K-Think) + Multi-Turn <think> Stripping",
                "avg_thinking_tokens": 960,
                "peak_active_kv_sessions": 820,
                "task_accuracy_pct": 94.0,
                "cost_per_task_usd": 0.0052,
                "recommended_scenario": "Standard production governance preserving full accuracy while cutting KV concurrency by 66.5%",
            },
            {
                "tier": "Tier 4 (Recommended): 3-Tier Difficulty Router + Dynamic </think> Budget Gate + Native MTP Speculative Decoding",
                "avg_thinking_tokens": 780,
                "peak_active_kv_sessions": 460,
                "task_accuracy_pct": 94.6,
                "cost_per_task_usd": 0.0038,
                "recommended_scenario": "Flagship reasoning platform architecture (-81.2% active KV slots & -78.9% cost per task)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json out of date")

    out_path.write_text(rendered, encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
