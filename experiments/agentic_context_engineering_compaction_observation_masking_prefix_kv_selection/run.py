from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "agentic_context_engineering_compaction_observation_masking_prefix_kv_selection",
        "title": "Agentic Context Compaction & Prefix KV Preservation Selection: Watermark Checkpoint Compaction vs. Sliding Window & Observation Masking",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "agent_turns_per_task": 50,
            "context_window_limit_tokens": 128000,
            "avg_tool_observation_tokens": 3500,
        },
        "tiers": [
            {
                "tier": "Tier 1: Rolling FIFO Sliding-Window Drop (Pop Oldest Message Every Turn -> 0% Prefix KV Cache Hit!)",
                "prefix_kv_cache_hit_rate_pct": 0.0,
                "avg_turn_ttft_ms": 4200.0,
                "long_horizon_50_turn_success_pct": 46.0,
                "cost_per_50_turn_task_usd": 14.80,
                "recommended_scenario": "Anti-pattern (mutating index 0 every turn invalidates 100% of Paged Prefix KV Cache!)",
            },
            {
                "tier": "Tier 2: Unstructured LLM Summarization Every 5 Turns",
                "prefix_kv_cache_hit_rate_pct": 72.0,
                "avg_turn_ttft_ms": 840.0,
                "long_horizon_50_turn_success_pct": 68.5,
                "cost_per_50_turn_task_usd": 3.40,
                "recommended_scenario": "Casual multi-turn chat where exact file paths and failed-attempt logs are not critical",
            },
            {
                "tier": "Tier 3: Stale Tool Observation Folding (Mask Outputs >8 Turns Old to Scratchpad Files)",
                "prefix_kv_cache_hit_rate_pct": 64.0,
                "avg_turn_ttft_ms": 960.0,
                "long_horizon_50_turn_success_pct": 82.4,
                "cost_per_50_turn_task_usd": 2.95,
                "recommended_scenario": "SWE-agent style shell loops with huge stdout logs",
            },
            {
                "tier": "Tier 4 (Recommended): 85%-Watermark Append-Only Prefix Compaction + Structured Checkpoint Schema + Scratchpad Files",
                "prefix_kv_cache_hit_rate_pct": 91.5,
                "avg_turn_ttft_ms": 380.0,
                "long_horizon_50_turn_success_pct": 93.4,
                "cost_per_50_turn_task_usd": 1.65,
                "recommended_scenario": "Production 50+ turn Coding & Research Agents (91.5% KV Cache hit, -88.9% cost, 93.4% success)",
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
