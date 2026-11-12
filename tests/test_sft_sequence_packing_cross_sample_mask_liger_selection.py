from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.sft_sequence_packing_cross_sample_mask_liger_selection.run import (
    evaluate_sft_packing_mask_liger_tiers,
)


class TestSftSequencePackingCrossSampleMaskLigerSelection(unittest.TestCase):
    def test_provenance_and_sft_packing_invariants(self) -> None:
        data = evaluate_sft_packing_mask_liger_tiers()
        self.assertEqual(
            data["experiment_id"],
            "sft_sequence_packing_cross_sample_mask_liger_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier1_throughput_speedup_x"], 5.53)
        self.assertEqual(summary["tier4_vs_tier1_logits_memory_reduction_x"], 32.3)
        self.assertEqual(summary["tier4_vs_tier2_accuracy_gain_pt"], 13.3)
        self.assertEqual(summary["tier4_padding_waste_pct"], 0.4)
        self.assertEqual(summary["tier4_vs_tier1_annual_savings_usd"], 525000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "sft_sequence_packing_cross_sample_mask_liger_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_sft_packing_mask_liger_tiers())


if __name__ == "__main__":
    unittest.main()
