from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "benchmark_id": "enterprise_rag_reranker_cascade_hybrid_bm25_colbert_cross_encoder_selection",
        "title": "Enterprise RAG Multi-Stage Retrieval & Reranking Cascade Architecture Selection Benchmark",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_assumptions": {
            "peak_rag_queries_per_sec": 500,
            "corpus_chunks_millions": 10,
            "stage1_hybrid_candidates": 100,
            "stage2_pruned_candidates": 25,
        },
        "tiers": [
            {
                "tier_id": "tier1_dense_vector_top5_only",
                "name": "Tier 1: Single-Stage Dense Vector Top-5 Only (No BM25, No Reranker)",
                "ndcg_at_5_pct": 71.2,
                "retrieval_and_rerank_p99_ms": 14.0,
                "llm_context_token_reduction_pct": 0.0,
                "monthly_compute_and_llm_token_tco_usd": 128000,
                "verdict": "Anti-Pattern: Misses 28.8% of exact clauses, SKUs & negated rules",
            },
            {
                "tier_id": "tier2_top25_unreranked_context_stuffing",
                "name": "Tier 2: Top-25 Unreranked Context Stuffing into Generator LLM",
                "ndcg_at_5_pct": 78.5,
                "retrieval_and_rerank_p99_ms": 22.0,
                "llm_context_token_reduction_pct": -400.0,
                "monthly_compute_and_llm_token_tco_usd": 185000,
                "verdict": "5x LLM prompt token bloat & 'Lost-in-the-Middle' distraction ($185K/mo)",
            },
            {
                "tier_id": "tier3_single_stage_heavy_reranker_top100",
                "name": "Tier 3: Heavy Cross-Encoder / Listwise LLM Over All Top-100 Candidates",
                "ndcg_at_5_pct": 95.4,
                "retrieval_and_rerank_p99_ms": 485.0,
                "llm_context_token_reduction_pct": 40.0,
                "monthly_compute_and_llm_token_tco_usd": 96000,
                "verdict": "485ms P99 latency bottleneck & requires 16x A100 GPUs to score 100x500 pairs/s",
            },
            {
                "tier_id": "tier4_three_stage_rrf_colbert_cross_encoder_cutoff",
                "name": "Tier 4: 3-Stage Cascade (100 RRF -> 25 FlashRank/ColBERT -> 5 TEI Cross-Encoder + Score>=0.35 Cutoff)",
                "ndcg_at_5_pct": 96.8,
                "retrieval_and_rerank_p99_ms": 42.0,
                "llm_context_token_reduction_pct": 62.0,
                "monthly_compute_and_llm_token_tco_usd": 19000,
                "verdict": "Recommended: 96.8% NDCG@5, 42ms P99 latency, -62% LLM prompt tokens & -89.7% TCO vs Tier 2",
            },
        ],
        "headline_findings": {
            "ndcg_at_5_gain_vs_tier1_pts": 25.6,
            "cascade_p99_latency_ms": 42.0,
            "llm_prompt_token_savings_pct": 62.0,
            "tco_savings_vs_tier2_pct": 89.7,
        },
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
            raise SystemExit("results.json does not exist")
        if out_path.read_text(encoding="utf-8") != rendered:
            raise SystemExit("results.json is out of date; re-run run.py")
        return 0

    out_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
