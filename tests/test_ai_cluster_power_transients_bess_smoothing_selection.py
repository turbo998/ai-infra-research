"""Unit tests for Round 57 AI Cluster Synchronous Power Transients (dP/dt) & BESS Smoothing Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ai_cluster_power_transients_bess_smoothing_selection.run import (
    evaluate_ai_cluster_power_transients_bess_matrix,
)


class TestAiClusterPowerTransientsBessSmoothingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ai_cluster_power_transients_bess_matrix()
        self.assertEqual(data["experiment_id"], "ai_cluster_power_transients_bess_smoothing_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier4_rack_supercap_cbu_firmware_ramp_substation_bess"]["grid_seen_power_swing_pct"],
            5.0,
        )
        self.assertEqual(
            tiers["tier4_rack_supercap_cbu_firmware_ramp_substation_bess"]["training_mfu_loss_pct"],
            0.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_swing_reduction_ratio"],
            20.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ai_cluster_power_transients_bess_smoothing_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ai_cluster_power_transients_bess_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
