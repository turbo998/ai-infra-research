from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.synthetic_data_distillation_pruning_pipeline.run import (
    evaluate_distillation_pruning_tiers,
)


class TestSyntheticDataDistillationPruningPipeline(unittest.TestCase):
    def test_provenance_and_distillation_invariants(self) -> None:
        data = evaluate_distillation_pruning_tiers()
        self.assertEqual(
            data["experiment_id"], "synthetic_data_distillation_pruning_pipeline"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier3_teacher_accuracy_retention_pct"], 96.3)
        self.assertEqual(summary["tier4_teacher_accuracy_retention_pct"], 98.8)
        self.assertEqual(summary["student_8b_vs_teacher_671b_serving_cost_saving_pct"], 94.4)
        self.assertEqual(summary["tier4_hallucination_rate_pct"], 1.8)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "synthetic_data_distillation_pruning_pipeline"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_distillation_pruning_tiers())


if __name__ == "__main__":
    unittest.main()
