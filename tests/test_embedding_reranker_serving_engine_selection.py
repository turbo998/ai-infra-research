from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.embedding_reranker_serving_engine_selection.run import (
    evaluate_embedding_reranker_tiers,
)


class TestEmbeddingRerankerServingEngineSelection(unittest.TestCase):
    def test_provenance_and_embedding_invariants(self) -> None:
        data = evaluate_embedding_reranker_tiers()
        self.assertEqual(
            data["experiment_id"], "embedding_reranker_serving_engine_selection"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier1_padding_waste_pct"], 82.4)
        self.assertEqual(summary["tier3_vs_tier1_throughput_gain_x"], 8.4)
        self.assertEqual(summary["tier4_vs_tier1_throughput_gain_x"], 14.8)
        self.assertEqual(summary["tier4_rerank50_p99_ms"], 14.2)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "embedding_reranker_serving_engine_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_embedding_reranker_tiers())


if __name__ == "__main__":
    unittest.main()
