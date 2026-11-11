"""Unit tests for Round 51 AI Cluster Day-0 Burn-In, Acceptance & Straggler Isolation Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_cluster_burnin_acceptance_dcgm_validation_selection.run import (
    evaluate_ai_cluster_burnin_acceptance_dcgm_matrix,
)


class TestAiClusterBurninAcceptanceDcgmValidationSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_cluster_burnin_acceptance_dcgm_matrix()
        self.assertEqual(data["experiment_id"], "ai_cluster_burnin_acceptance_dcgm_validation_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_48h_thermal_shock_superbench_preflight_gate"]["defect_detection_rate_pct"],
            98.0,
        )
        self.assertLess(
            tiers["tier3_hierarchical_nccl_bisection_pairwise_search"]["straggler_localization_minutes"],
            5.0,
        )
        self.assertGreater(
            data["summary"]["tier4_month1_savings_vs_tier1_usd"],
            350000.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_cluster_burnin_acceptance_dcgm_validation_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_cluster_burnin_acceptance_dcgm_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
