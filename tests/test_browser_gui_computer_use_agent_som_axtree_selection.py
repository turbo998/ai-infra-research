"""Unit tests for Round 68 Enterprise Browser & GUI Computer-Use Agent Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.browser_gui_computer_use_agent_som_axtree_selection.run import (
    evaluate_browser_gui_computer_use_agent_matrix,
)


class TestBrowserGuiComputerUseAgentSomAxtreeSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_browser_gui_computer_use_agent_matrix()
        self.assertEqual(
            data["experiment_id"],
            "browser_gui_computer_use_agent_som_axtree_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_hybrid_cdp_api_bypass_plus_axtree_som_session_pool"]["workflow_success_rate_pct"],
            90.0,
        )
        self.assertLess(
            tiers["tier3_pruned_axtree_plus_set_of_marks_som_grounding"]["click_grounding_error_pct"],
            5.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_token_reduction_ratio"],
            50.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "browser_gui_computer_use_agent_som_axtree_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_browser_gui_computer_use_agent_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
