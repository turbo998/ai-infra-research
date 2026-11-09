from __future__ import annotations

import json
from pathlib import Path


def evaluate_enterprise_rag_tiers() -> dict:
    """Deterministic evaluator for 4 Enterprise RAG & GraphRAG Knowledge Indexing Tiers.

    Workload Baseline:
      - Enterprise Corpus: 50,000 Regulatory, Financial & Technical Documents (10.0 Million Tokens total)
      - Daily Incremental Document Updates: 1,500 documents / day
      - Query Mix:
        1) Single-Hop Exact Factoid / Clause Lookup ("What is the penalty clause in Contract #A-492?")
        2) Cross-Paragraph Pronoun / Implicit-Context Query ("When did its Q3 margin drop below 12%?")
        3) Global Thematic & 3-Hop Relational Synthesis ("What are the top systemic supply-chain risks across all 50,000 filings?")
      - Reference Eager Full GraphRAG baseline included for cost comparison ($4,850 per 10M tokens).
    """
    tiers = [
        {
            "tier_id": "tier1_naive_512tok_dense_vector_rag",
            "name": "Tier 1: Naive 512-Token Fixed Chunking + Dense Vector Top-K",
            "name_zh": "Tier 1: 传统 512-Token 硬切片 + 纯稠密向量 Top-K 检索",
            "indexing_mechanism": "Sliding 512-token window + Single Dense Embedding (bge-large / text-embedding-3)",
            "single_hop_factoid_accuracy_pct": 71.5,
            "implicit_context_pronoun_accuracy_pct": 54.0,
            "global_multihop_synthesis_accuracy_pct": 24.5,
            "blended_enterprise_accuracy_pct": 50.0,
            "indexing_cost_per_10m_tokens_usd": 14.0,
            "query_p95_latency_ms": 145.0,
            "incremental_update_latency_sec": 0.8,
        },
        {
            "tier_id": "tier2_hybrid_bm25_dense_cross_encoder_rerank",
            "name": "Tier 2: Hybrid BM25 + Dense Vector (RRF) + Cross-Encoder Reranker (bge-reranker-v2)",
            "name_zh": "Tier 2: 混合检索 (BM25 + 向量 RRF) + Cross-Encoder 精排",
            "indexing_mechanism": "Sparse SPLADE/BM25 + Dense Embedding + Top-50 -> Top-5 Cross-Encoder Rerank",
            "single_hop_factoid_accuracy_pct": 91.2,
            "implicit_context_pronoun_accuracy_pct": 68.5,
            "global_multihop_synthesis_accuracy_pct": 38.5,
            "blended_enterprise_accuracy_pct": 66.1,
            "indexing_cost_per_10m_tokens_usd": 18.0,
            "query_p95_latency_ms": 235.0,
            "incremental_update_latency_sec": 1.2,
        },
        {
            "tier_id": "tier3_contextual_late_chunking_hybrid_rag",
            "name": "Tier 3: Contextual Retrieval (Prefix-Cached Context Header / Late-Chunking) + Hybrid Rerank",
            "name_zh": "Tier 3: 上下文增强切片 (Contextual Retrieval / Late-Chunking) + 混合精排",
            "indexing_mechanism": "Document-Level KV Prefix Context Header (or Long-Context Late Pooling) + BM25 + Dense + Rerank",
            "single_hop_factoid_accuracy_pct": 94.6,
            "implicit_context_pronoun_accuracy_pct": 91.8,
            "global_multihop_synthesis_accuracy_pct": 52.0,
            "blended_enterprise_accuracy_pct": 79.5,
            "indexing_cost_per_10m_tokens_usd": 42.0,
            "query_p95_latency_ms": 245.0,
            "incremental_update_latency_sec": 2.5,
        },
        {
            "tier_id": "tier4_lazy_graphrag_hybrid_router",
            "name": "Tier 4: Adaptive Router -> Contextual Hybrid RAG + LazyGraphRAG / LightRAG (Concept Subgraph + On-Demand Community Synthesis)",
            "name_zh": "Tier 4: 自适应路由 + 上下文混合检索 + LazyGraphRAG/LightRAG 按需子图归纳",
            "indexing_mechanism": "NLP Noun-Phrase Concept Co-occurrence Graph + Leiden Hierarchy + Query-Time Iterative Subgraph Map-Reduce",
            "single_hop_factoid_accuracy_pct": 94.8,
            "implicit_context_pronoun_accuracy_pct": 92.4,
            "global_multihop_synthesis_accuracy_pct": 88.5,
            "blended_enterprise_accuracy_pct": 91.9,
            "indexing_cost_per_10m_tokens_usd": 38.0,
            "query_p95_latency_ms": 680.0,
            "incremental_update_latency_sec": 3.2,
        },
    ]

    # Reference comparison with Legacy Eager Full GraphRAG (LLM extracts every entity/edge + pre-summarizes all Leiden communities)
    graphrag_indexing_comparison = {
        "eager_full_graphrag_index_cost_per_10m_usd": 4850.0,
        "eager_full_graphrag_global_accuracy_pct": 89.4,
        "eager_full_graphrag_query_p95_ms": 4200.0,
        "lazy_graphrag_index_cost_per_10m_usd": 38.0,
        "lazy_graphrag_global_accuracy_pct": 88.5,
        "lazy_graphrag_query_p95_ms": 680.0,
    }

    t1 = tiers[0]
    t2 = tiers[1]
    t3 = tiers[2]
    t4 = tiers[3]

    summary = {
        "tier3_single_hop_factoid_accuracy_pct": t3["single_hop_factoid_accuracy_pct"],
        "tier3_vs_tier1_implicit_context_gain_pt": round(
            t3["implicit_context_pronoun_accuracy_pct"] - t1["implicit_context_pronoun_accuracy_pct"], 1
        ),
        "tier4_vs_tier2_global_multihop_gain_pt": round(
            t4["global_multihop_synthesis_accuracy_pct"] - t2["global_multihop_synthesis_accuracy_pct"], 1
        ),
        "tier4_vs_eager_graphrag_index_cost_saving_pct": round(
            (
                graphrag_indexing_comparison["eager_full_graphrag_index_cost_per_10m_usd"]
                - graphrag_indexing_comparison["lazy_graphrag_index_cost_per_10m_usd"]
            )
            / graphrag_indexing_comparison["eager_full_graphrag_index_cost_per_10m_usd"]
            * 100.0,
            1,
        ),
        "tier4_query_p95_latency_ms": t4["query_p95_latency_ms"],
    }

    return {
        "experiment_id": "enterprise_rag_graphrag_hybrid_selection",
        "date": "2026-11-09",
        "provenance": {
            "type": "synthetic_rag_chunking_contextual_and_lazy_graphrag_cost_accuracy_model_calibrated_to_anonymized_reported_specs",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "Anthropic Contextual Retrieval (Prompt-Cached Document Header + BM25 + Reranker) Failure-Rate Reduction Model",
                "Jina AI Late-Chunking (Long-Context Transformer Token Pooling Prior to Chunking)",
                "Microsoft Research LazyGraphRAG & HKU LightRAG (Noun-Phrase Concept Graph vs Eager Full LLM Graph Extraction Cost/Quality)",
            ],
        },
        "workload_baseline": {
            "corpus_documents": 50000,
            "corpus_total_tokens": 10000000,
            "daily_incremental_updates_docs": 1500,
        },
        "tiers": tiers,
        "graphrag_indexing_comparison": graphrag_indexing_comparison,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_enterprise_rag_tiers()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
