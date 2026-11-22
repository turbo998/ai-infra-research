from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: Raw 500-Table DDL Stuffing + 70B General LLM",
            "business_sql_accuracy_pct": 38.5,
            "prompt_tokens_per_query": 95000,
            "p95_e2e_latency_ms": 6800.0,
            "monthly_runaway_warehouse_scan_usd": 42000,
            "required_h200_gpus_at_30qps": 48,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: Vector Top-15 Raw DDL Retrieval + 70B LLM (No MDL)",
            "business_sql_accuracy_pct": 64.2,
            "prompt_tokens_per_query": 4500,
            "p95_e2e_latency_ms": 1850.0,
            "monthly_runaway_warehouse_scan_usd": 18500,
            "required_h200_gpus_at_30qps": 12,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: Multi-Turn Self-Debug SQL Agent (5 Live DB Retries)",
            "business_sql_accuracy_pct": 86.5,
            "prompt_tokens_per_query": 18500,
            "p95_e2e_latency_ms": 5400.0,
            "monthly_runaway_warehouse_scan_usd": 9200,
            "required_h200_gpus_at_30qps": 20,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: Semantic Metric Layer (MDL) + FK-Subgraph + AST/EXPLAIN Gate + 32B/14B SQL Pool (Recommended)",
            "business_sql_accuracy_pct": 95.4,
            "prompt_tokens_per_query": 2400,
            "p95_e2e_latency_ms": 620.0,
            "monthly_runaway_warehouse_scan_usd": 0,
            "required_h200_gpus_at_30qps": 4,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_text_to_sql_semantic_layer_mdl_ast_sandbox_sizing_selection",
        "title": "Enterprise Text-to-SQL & Conversational BI (Semantic Metric Layer MDL + FK-Subgraph + AST Gate) Sizer (#499)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "warehouse_baseline_30qps": {
            "total_warehouse_tables": 500,
            "total_columns": 12000,
            "accuracy_gain_pts_vs_raw_ddl": round(95.4 - 38.5, 1),
            "token_reduction_pct": round((1.0 - 2400 / 95000) * 100, 1),
            "gpu_saving_pct": round((1.0 - 4 / 48) * 100, 1),
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
