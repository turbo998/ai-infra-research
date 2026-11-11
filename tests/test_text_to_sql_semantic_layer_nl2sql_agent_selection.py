"""Unit tests for Round 64 Enterprise Text-to-SQL (NL2SQL), LSH Value Linking & Governed Semantic Layer Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.text_to_sql_semantic_layer_nl2sql_agent_selection.run import (
    evaluate_text_to_sql_semantic_layer_matrix,
)


class TestTextToSqlSemanticLayerNl2SqlAgentSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_text_to_sql_semantic_layer_matrix()
        self.assertEqual(
            data["experiment_id"],
            "text_to_sql_semantic_layer_nl2sql_agent_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_governed_semantic_layer_dsl_plus_lsh_sql_hybrid"]["execution_accuracy_ex_pct"],
            90.0,
        )
        self.assertLess(
            tiers["tier4_governed_semantic_layer_dsl_plus_lsh_sql_hybrid"]["silent_kpi_formula_drift_pct"],
            1.5,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_kpi_drift_reduction_ratio"],
            30.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "text_to_sql_semantic_layer_nl2sql_agent_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_text_to_sql_semantic_layer_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
