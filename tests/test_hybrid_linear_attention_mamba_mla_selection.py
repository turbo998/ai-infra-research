from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.hybrid_linear_attention_mamba_mla_selection.run import (
    evaluate_attention_state_space_tiers,
)


class TestHybridLinearAttentionMambaMlaSelection(unittest.TestCase):
    def test_provenance_and_attention_invariants(self) -> None:
        data = evaluate_attention_state_space_tiers()
        self.assertEqual(
            data["experiment_id"], "hybrid_linear_attention_mamba_mla_selection"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier3_vs_tier1_kv_compression_x"], 5.69)
        self.assertEqual(summary["tier4_vs_tier1_kv_compression_x"], 3.85)
        self.assertEqual(summary["tier4_vs_tier1_128k_decode_gain_x"], 3.65)
        self.assertEqual(summary["tier4_ruler_128k_needle_accuracy_pct"], 96.4)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "hybrid_linear_attention_mamba_mla_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_attention_state_space_tiers())


if __name__ == "__main__":
    unittest.main()
