"""Unit tests for Round 47 Test-Time Compute Scaling & Long-CoT PRM Routing Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.test_time_compute_long_cot_routing_selection.run import (
    evaluate_test_time_compute_long_cot_routing_matrix,
)


class TestTestTimeComputeLongCotRoutingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_test_time_compute_long_cot_routing_matrix()
        self.assertEqual(data["experiment_id"], "test_time_compute_long_cot_routing_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier1_unbounded_best_of_16_brute_force"]["avg_tokens_per_query"],
            tiers["tier4_difficulty_router_plus_radix_shared_tree_prm"]["avg_tokens_per_query"] * 25.0,
        )
        self.assertGreater(
            tiers["tier4_difficulty_router_plus_radix_shared_tree_prm"]["reasoning_accuracy_pct"],
            tiers["tier1_unbounded_best_of_16_brute_force"]["reasoning_accuracy_pct"],
        )
        self.assertGreater(
            data["summary"]["tier4_cost_reduction_vs_tier1_pct"],
            95.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "test_time_compute_long_cot_routing_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_test_time_compute_long_cot_routing_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
