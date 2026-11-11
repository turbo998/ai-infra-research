"""Unit tests for Round 62 AI Cluster Grey Failure / Runtime Straggler Telemetry & Mitigation Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_cluster_straggler_grey_failure_mitigation_selection.run import (
    evaluate_ai_cluster_straggler_grey_failure_matrix,
)


class TestAiClusterStragglerGreyFailureMitigationSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_cluster_straggler_grey_failure_matrix()
        self.assertEqual(
            data["experiment_id"],
            "ai_cluster_straggler_grey_failure_mitigation_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_predictive_telemetry_plus_live_rdma_warm_standby_swap"]["effective_cluster_goodput_pct"],
            97.0,
        )
        self.assertLess(
            tiers["tier4_predictive_telemetry_plus_live_rdma_warm_standby_swap"]["straggler_attribution_time_s"],
            3.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_waste_reduction_ratio"],
            20.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_cluster_straggler_grey_failure_mitigation_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_cluster_straggler_grey_failure_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
