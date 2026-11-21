from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "enterprise_rag_search_azure_ai_search_milvus_qdrant_pgvector_tco_selection",
        "title": "Enterprise Production RAG & Billion-Scale Vector Search Engine Selection Guide: Azure AI Search vs. Milvus 2.5 (DiskANN/CAGRA) vs. Qdrant vs. PostgreSQL pgvectorscale",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "vector_count_millions": 100,
            "embedding_dimension": 1536,
            "query_qps": 200,
        },
        "tiers": [
            {
                "tier": "Tier 1: Uncompressed In-Memory FP32 HNSW + Dense-Only Retrieval (No BM25 or Reranker)",
                "cluster_ram_gb": 896,
                "p99_search_latency_ms": 14.2,
                "recall_at_5_pct": 71.5,
                "monthly_infra_tco_usd": 9800,
                "recommended_scenario": "Obsolete at >=50M vectors due to 10x RAM bloat and poor exact-keyword recall",
            },
            {
                "tier": "Tier 2: HNSW SQ8 Quantization + Naive Post-Retrieval ACL Filtering",
                "cluster_ram_gb": 240,
                "p99_search_latency_ms": 16.8,
                "recall_at_5_pct": 78.2,
                "monthly_infra_tco_usd": 3600,
                "recommended_scenario": "Post-filtering causes empty top-K results for restricted-permission users",
            },
            {
                "tier": "Tier 3: Turnkey Azure AI Search (Compressed Vector + BM25 RRF + L2 Semantic Ranker + Entra ACL)",
                "cluster_ram_gb": 128,
                "p99_search_latency_ms": 28.5,
                "recall_at_5_pct": 94.2,
                "monthly_infra_tco_usd": 2850,
                "recommended_scenario": "Best fit for M365/Azure enterprise RAG needing zero-ops document ACL security trimming",
            },
            {
                "tier": "Tier 4 (High-Scale Recommended): NVMe DiskANN / BQ + Hybrid BM25 + Bitmap Pre-ACL + Cross-Encoder Rerank",
                "cluster_ram_gb": 72,
                "p99_search_latency_ms": 21.4,
                "recall_at_5_pct": 94.8,
                "monthly_infra_tco_usd": 1800,
                "recommended_scenario": "100M-to-10B vector gold standard (Milvus 2.5 / Qdrant / pgvectorscale) cutting RAM by 92.0% and TCO by 81.6%",
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
