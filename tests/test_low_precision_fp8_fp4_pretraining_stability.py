from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.low_precision_fp8_fp4_pretraining_stability.run import (
    evaluate_low_precision_pretraining_tiers,
)


class TestLowPrecisionFp8Fp4PretrainingStability(unittest.TestCase):
    def test_provenance_and_pretraining_invariants(self) -> None:
        data = evaluate_low_precision_pretraining_tiers()
        self.assertEqual(
            data["experiment_id"], "low_precision_fp8_fp4_pretraining_stability"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier3_vs_tier1_throughput_gain_x"], 1.48)
        self.assertEqual(summary["tier3_loss_rel_error_pct"], 0.07)
        self.assertEqual(summary["tier4_vs_tier1_throughput_gain_x"], 2.15)
        self.assertEqual(summary["tier3_vs_tier1_compute_cost_savings_usd"], 1490000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "low_precision_fp8_fp4_pretraining_stability"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_low_precision_pretraining_tiers())


if __name__ == "__main__":
    unittest.main()
