from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.offline_batch_llm_inference_spot_prefix_bucketing_selection.run import (
    evaluate_offline_batch_inference_spot_bucketing_tiers,
)


class TestOfflineBatchLlmInferenceSpotPrefixBucketingSelection(unittest.TestCase):
    def test_provenance_and_offline_batch_invariants(self) -> None:
        data = evaluate_offline_batch_inference_spot_bucketing_tiers()
        self.assertEqual(
            data["experiment_id"],
            "offline_batch_llm_inference_spot_prefix_bucketing_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier1_throughput_gain_x"], 3.98)
        self.assertEqual(summary["tier4_vs_tier1_cost_reduction_pct"], 90.9)
        self.assertEqual(summary["tier4_vs_tier1_annual_savings_usd"], 7748000)
        self.assertEqual(summary["tier3_vs_tier1_prefix_hit_gain_pt"], 82.7)
        self.assertEqual(summary["tier4_spot_lost_work_pct"], 0.8)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "offline_batch_llm_inference_spot_prefix_bucketing_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            on_disk, evaluate_offline_batch_inference_spot_bucketing_tiers()
        )


if __name__ == "__main__":
    unittest.main()
