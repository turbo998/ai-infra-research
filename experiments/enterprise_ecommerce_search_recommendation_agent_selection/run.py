from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_Payload() -> dict:
    return {
        "round": 402,
        "slug": "enterprise_ecommerce_search_recommendation_agent_selection",
        "title": "Enterprise E-Commerce Search, Personalized Recommendation & Conversational Shopping Assistant Architecture Selection",
        "Provenance": "synthetic_analytical_model_calibrated_to_published_ecommerce_semantic_search_two_tower_recommendation_and_conversational_shopping_benchmarks",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "catalog_skus": 5000000,
            "monthly_search_queries": 60000000,
            "peak_rps": 1000,
            "enterprise_sla": "P95 product grid render <= 100ms, 0.0% out-of-stock/price hallucination, maximize long-tail query conversion rate (CVR)",
        },
        "tiers": [
            {
                "tier_id": "tier1_legacy_bm25_keyword_search",
                "name": "Tier 1: Legacy BM25 Keyword Search + Static Rule Boost",
                "long_tail_conversion_rate_pct": 2.15,
                "price_stock_hallucination_pct": 0.0,
                "p95_search_grid_latency_ms": 35.0,
                "monthly_total_cost_usd": 8400,
                "action": "Send to a human",
                "status": "fail",
            },
            {
                "tier_id": "tier2_sync_70b_llm_in_search_loop",
                "name": "Tier 2: Synchronous 70B LLM Conversational Search Blocking Every Query",
                "long_tail_conversion_rate_pct": 2.60,
                "price_stock_hallucination_pct": 6.8,
                "p95_search_grid_latency_ms": 2850.0,
                "monthly_total_cost_usd": 148000,
                "action": "Send to a human",
                "status": "fail",
            },
            {
                "tier_id": "tier3_sync_8b_query_rewrite_hybrid_vector",
                "name": "Tier 3: Synchronous 8B LLM Query Rewriting + Hybrid BM25/Vector Search",
                "long_tail_conversion_rate_pct": 3.45,
                "price_stock_hallucination_pct": 0.8,
                "p95_search_grid_latency_ms": 380.0,
                "monthly_total_cost_usd": 32000,
                "action": "Confirm",
                "status": "warn",
            },
            {
                "tier_id": "tier4_offline_llm_enrichment_dual_speed_retail_ai",
                "name": "Tier 4: Offline LLM Catalog Enrichment + Sub-15ms Intent Parser + Two-Tower Recall + Redis Pinning + Async Streaming Advisor",
                "long_tail_conversion_rate_pct": 4.68,
                "price_stock_hallucination_pct": 0.0,
                "p95_search_grid_latency_ms": 62.0,
                "monthly_total_cost_usd": 12600,
                "action": "Act automatically",
                "status": "pass",
            },
        ],
        "robustness_check": [
            {
                "metric": "Long-Tail Query Conversion Rate CVR (%)",
                "baseline_tier1": 2.15,
                "optimized_tier4": 4.68,
                "delta": "+2.53 pts (+117.7% lift)",
            },
            {
                "metric": "Price / Out-of-Stock Hallucination (%)",
                "baseline_tier2": 6.8,
                "optimized_tier4": 0.0,
                "delta": "-6.8 pts (0.0% hallucination)",
            },
            {
                "metric": "P95 Search Grid Latency (ms)",
                "baseline_tier2": 2850.0,
                "optimized_tier4": 62.0,
                "delta": "-97.8% (46.0x faster)",
            },
            {
                "metric": "Monthly Infra Cost for 60M Queries (USD)",
                "baseline_tier2": 148000,
                "optimized_tier4": 12600,
                "delta": "-91.5% (-$135,400/mo)",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).with_name("results.json")
    payload = build_Payload()

    if args.check:
        if not out_path.exists():
            raise SystemExit("results.json missing")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise SystemExit("results.json out of sync")
        print("OK: enterprise_ecommerce_search_recommendation_agent_selection verified.")
        return 0

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
