from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.batch_invariant_deterministic_llm_inference_selection.run import (
    evaluate_batch_invariant_deterministic_inference_tiers,
)


class TestBatchInvariantDeterministicLlmInferenceSelection(unittest.TestCase):
    def test_provenance_and_batch_invariant_invariants(self) -> None:
        data = evaluate_batch_invariant_deterministic_inference_tiers()
        self.assertEqual(
            data["experiment_id"],
            "batch_invariant_deterministic_llm_inference_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_exact_match_pct"], 100.0)
        self.assertEqual(summary["tier1_exact_match_pct"], 61.5)
        self.assertEqual(summary["tier4_vs_tier2_throughput_speedup_x"], 11.16)
        self.assertEqual(summary["tier4_vs_tier1_throughput_retention_pct"], 93.0)
        self.assertEqual(summary["tier4_vs_tier2_annual_savings_usd"], 1480000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "batch_invariant_deterministic_llm_inference_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            on_disk, evaluate_batch_invariant_deterministic_inference_tiers()
        )


if __name__ == "__main__":
    unittest.main()
