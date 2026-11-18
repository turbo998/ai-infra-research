"""Unit tests for Distributed Multimodal Data Processing & Ray Data Heterogeneous CPU-GPU Streaming Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.ray_data_heterogeneous_cpu_gpu_streaming_etl_selection.run import (
    evaluate_ray_data_heterogeneous_cpu_gpu_streaming_etl_matrix,
)


class TestRayDataHeterogeneousCpuGpuStreamingEtlSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_ray_data_heterogeneous_cpu_gpu_streaming_etl_matrix()
        self.assertEqual(
            data["experiment_id"],
            "ray_data_heterogeneous_cpu_gpu_streaming_etl_selection",
        )
        self.assertEqual(data["round"], 180)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertEqual(
            tiers["tier4_ray_data_streaming_backpressure_elastic_spot_cpu"]["intermediate_storage_tb"],
            0.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_gpu_util_gain_pts"],
            40.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_cost_reduction_pct"],
            60.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "ray_data_heterogeneous_cpu_gpu_streaming_etl_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_ray_data_heterogeneous_cpu_gpu_streaming_etl_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
