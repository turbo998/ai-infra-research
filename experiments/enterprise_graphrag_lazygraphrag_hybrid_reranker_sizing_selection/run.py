from __future__ import annotations

import json
from pathlib import Path


def build_results() -> dict:
    tiers = [
        {
            "tier": "Tier 1: 128K Long-Context Stuffing (No Retrieval Index)",
            "factoid_accuracy_pct": 88.0,
            "global_multihop_accuracy_pct": 72.0,
            "index_build_cost_500m_tokens_usd": 0,
            "p95_query_latency_ms": 3850.0,
            "cost_per_1000_queries_usd": 185.0,
            "required_h200_gpus_at_50qps": 64,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 2: Naive Full LLM GraphRAG (Map-Reduce on Every Query)",
            "factoid_accuracy_pct": 89.5,
            "global_multihop_accuracy_pct": 93.8,
            "index_build_cost_500m_tokens_usd": 4500,
            "p95_query_latency_ms": 14200.0,
            "cost_per_1000_queries_usd": 420.0,
            "required_h200_gpus_at_50qps": 96,
            "action": "Send to a human",
        },
        {
            "tier": "Tier 3: Pure Hybrid Vector + BM25 + Cross-Encoder Reranker",
            "factoid_accuracy_pct": 94.5,
            "global_multihop_accuracy_pct": 58.0,
            "index_build_cost_500m_tokens_usd": 18,
            "p95_query_latency_ms": 420.0,
            "cost_per_1000_queries_usd": 4.2,
            "required_h200_gpus_at_50qps": 4,
            "action": "Confirm",
        },
        {
            "tier": "Tier 4: Intent-Routed 85% Hybrid Reranker + 15% LazyGraphRAG (Recommended)",
            "factoid_accuracy_pct": 94.8,
            "global_multihop_accuracy_pct": 93.2,
            "index_build_cost_500m_tokens_usd": 128,
            "p95_query_latency_ms": 680.0,
            "cost_per_1000_queries_usd": 8.5,
            "required_h200_gpus_at_50qps": 6,
            "action": "Act automatically",
        },
    ]

    return {
        "experiment_id": "enterprise_graphrag_lazygraphrag_hybrid_reranker_sizing_selection",
        "title": "Enterprise GraphRAG / LazyGraphRAG vs. Hybrid Vector+Reranker vs. 128K Stuffing Sizer (#497)",
        "provenance": "synthetic + anonymized_reported (independently_reproduced=false)",
        "corpus_500m_tokens_50qps": {
            "corpus_tokens_millions": 500,
            "index_cost_saving_pct_vs_full_graphrag": round((1.0 - 128 / 4500) * 100, 1),
            "query_speedup_vs_full_graphrag": round(14200.0 / 680.0, 1),
            "gpu_saving_pct_vs_full_graphrag": round((1.0 - 6 / 96) * 100, 1),
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
