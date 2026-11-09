from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.k8s_serverless_gpu_autoscaling_coldstart.run import (
    evaluate_serverless_gpu_coldstart_tiers,
)


class TestK8sServerlessGpuAutoscalingColdstart(unittest.TestCase):
    def test_provenance_and_coldstart_invariants(self) -> None:
        data = evaluate_serverless_gpu_coldstart_tiers()
        self.assertEqual(
            data["experiment_id"], "k8s_serverless_gpu_autoscaling_coldstart"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier1_total_coldstart_sec"], 285.0)
        self.assertEqual(summary["tier3_total_coldstart_sec"], 4.6)
        self.assertEqual(summary["tier4_total_coldstart_sec"], 1.8)
        self.assertEqual(summary["tier4_vs_tier1_speedup_x"], 158.33)
        self.assertEqual(summary["tier4_monthly_gpu_cost_saving_pct"], 68.4)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "k8s_serverless_gpu_autoscaling_coldstart"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_serverless_gpu_coldstart_tiers())


if __name__ == "__main__":
    unittest.main()
