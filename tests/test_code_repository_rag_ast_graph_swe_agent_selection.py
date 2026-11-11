"""Unit tests for Round 61 Enterprise Codebase Indexing, Tree-sitter AST Graph & SWE-Agent Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.code_repository_rag_ast_graph_swe_agent_selection.run import (
    evaluate_code_repository_rag_ast_swe_agent_matrix,
)


class TestCodeRepositoryRagAstGraphSweAgentSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_code_repository_rag_ast_swe_agent_matrix()
        self.assertEqual(
            data["experiment_id"],
            "code_repository_rag_ast_graph_swe_agent_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_scip_symbol_graph_pagerank_repomap_ast_agent"]["cross_file_localization_recall_pct"],
            90.0,
        )
        self.assertGreater(
            tiers["tier4_scip_symbol_graph_pagerank_repomap_ast_agent"]["swe_bench_multifile_resolve_pct"],
            65.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_token_reduction_ratio"],
            7.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "code_repository_rag_ast_graph_swe_agent_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_code_repository_rag_ast_swe_agent_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
