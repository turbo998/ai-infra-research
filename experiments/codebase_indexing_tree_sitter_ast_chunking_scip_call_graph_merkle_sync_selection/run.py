from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "experiment_id": "codebase_indexing_tree_sitter_ast_chunking_scip_call_graph_merkle_sync_selection",
        "title": "Enterprise Codebase Indexing & Repo Retrieval Selection: Tree-sitter AST Chunking + SCIP Call-Graph + Merkle Sync vs. Naive Line Split",
        "provenance": "synthetic + anonymized_reported",
        "independently_reproduced": False,
        "physical_gpu_cluster_measurement": "not_measured",
        "workload_basis": {
            "monorepo_files": 50000,
            "lines_of_code_millions": 15.0,
            "developer_seats": 1000,
            "eval_benchmark": "SWE-bench Verified File & Function Localization Recall@10",
        },
        "tiers": [
            {
                "tier": "Tier 1: Naive Fixed-Token / Line Splitting (512 Tokens) + Full Branch Re-Embedding",
                "swe_localization_recall_at_10_pct": 48.5,
                "branch_switch_sync_latency_sec": 2520.0,
                "embedding_compute_reuse_pct": 0.0,
                "monthly_indexing_gpu_cost_usd": 48000,
                "recommended_scenario": "Anti-pattern (slices functions mid-block; re-indexes all 50k files on git checkout)",
            },
            {
                "tier": "Tier 2: Tree-sitter AST Semantic Chunking (cAST) + Code Embedding Only",
                "swe_localization_recall_at_10_pct": 74.2,
                "branch_switch_sync_latency_sec": 45.0,
                "embedding_compute_reuse_pct": 85.0,
                "monthly_indexing_gpu_cost_usd": 7200,
                "recommended_scenario": "Small single-branch repositories with natural-language queries",
            },
            {
                "tier": "Tier 3: Tree-sitter cAST (With Parent Scope Header) + Hybrid Trigram/BM25 + Merkle-Tree Sync",
                "swe_localization_recall_at_10_pct": 84.6,
                "branch_switch_sync_latency_sec": 0.18,
                "embedding_compute_reuse_pct": 99.8,
                "monthly_indexing_gpu_cost_usd": 1450,
                "recommended_scenario": "Fast IDE workspace indexing across 1,000 developer branches (180ms Merkle diff)",
            },
            {
                "tier": "Tier 4 (Recommended): Tree-sitter cAST + Hybrid Trigram/Vector + SCIP/LSP 1-Hop Call-Graph Expansion + Merkle Sync",
                "swe_localization_recall_at_10_pct": 94.8,
                "branch_switch_sync_latency_sec": 0.18,
                "embedding_compute_reuse_pct": 99.8,
                "monthly_indexing_gpu_cost_usd": 1680,
                "recommended_scenario": "Enterprise monorepo Coding Agents, PR Reviewers & multi-file bug localization",
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
