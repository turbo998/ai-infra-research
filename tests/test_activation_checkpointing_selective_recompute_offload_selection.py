from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.activation_checkpointing_selective_recompute_offload_selection.run import (
    evaluate_activation_checkpointing_offload_tiers,
)


class TestActivationCheckpointingSelectiveRecomputeOffloadSelection(unittest.TestCase):
    def test_provenance_and_sac_offload_invariants(self) -> None:
        data = evaluate_activation_checkpointing_offload_tiers()
        self.assertEqual(
            data["experiment_id"],
            "activation_checkpointing_selective_recompute_offload_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier1_activation_hbm_reduction_pct"], 91.2)
        self.assertEqual(summary["tier4_vs_tier2_goodput_speedup_x"], 1.31)
        self.assertEqual(summary["tier4_vs_tier2_mfu_gain_pt"], 13.6)
        self.assertEqual(summary["tier4_recompute_flop_overhead_pct"], 2.8)
        self.assertEqual(summary["tier4_vs_tier2_annual_savings_usd"], 612000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "activation_checkpointing_selective_recompute_offload_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_activation_checkpointing_offload_tiers())


if __name__ == "__main__":
    unittest.main()
