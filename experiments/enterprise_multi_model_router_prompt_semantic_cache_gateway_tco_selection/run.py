from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "enterprise_multi_model_router_prompt_semantic_cache_gateway_tco_selection",
        "title": "Enterprise Multi-Model Cascade Router & 3-Tier Caching Gateway Selection Guide: Azure APIM vs. Envoy AI Gateway (K8s GIE) vs. LiteLLM vs. Higress",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "daily_requests_millions": 50.0,
            "traffic_composition": "78% Fast/Medium Tasks (SLM 8B-32B Eligible) + 22% Complex Reasoning Tasks (Frontier 671B/GPT-4o)",
        },
        "tiers": [
            {
                "tier": "Tier 1: Unrouted 100% Frontier Model Calls Without Caching",
                "semantic_false_positive_rate_pct": 0.0,
                "p95_ttft_ms": 1120.0,
                "task_accuracy_pct": 95.4,
                "cost_per_1m_requests_usd": 4850,
                "recommended_scenario": "Baseline unoptimized deployment paying frontier prices for trivial extraction/classification",
            },
            {
                "tier": "Tier 2: Naive Cosine-Only Semantic Cache (Threshold=0.90 Without Entity/Number Verification)",
                "semantic_false_positive_rate_pct": 8.2,
                "p95_ttft_ms": 640.0,
                "task_accuracy_pct": 86.1,
                "cost_per_1m_requests_usd": 2920,
                "recommended_scenario": "Severe anti-pattern returning wrong answers on numeric/date/negation variations",
            },
            {
                "tier": "Tier 3: 78/22 Multi-Model Cascade Router + Exact Engine Prefix KV Caching",
                "semantic_false_positive_rate_pct": 0.0,
                "p95_ttft_ms": 310.0,
                "task_accuracy_pct": 95.1,
                "cost_per_1m_requests_usd": 1180,
                "recommended_scenario": "Safe production baseline cutting cost by 75.7% with zero semantic collision risk",
            },
            {
                "tier": "Tier 4 (Recommended): 78/22 Cascade Router + Entity-Guarded Semantic Cache (FAQ) + K8s GIE Prefix KV Routing",
                "semantic_false_positive_rate_pct": 0.01,
                "p95_ttft_ms": 195.0,
                "task_accuracy_pct": 95.5,
                "cost_per_1m_requests_usd": 850,
                "recommended_scenario": "Flagship enterprise AI gateway stack (-82.5% cost per 1M requests & 5.74x faster P95 TTFT)",
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
