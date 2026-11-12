from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.async_tensor_parallelism_gemm_comm_overlap_selection.run import (
    evaluate_async_tensor_parallelism_overlap_tiers,
)


class TestAsyncTensorParallelismGemmCommOverlapSelection(unittest.TestCase):
    def test_provenance_and_async_tp_invariants(self) -> None:
        data = evaluate_async_tensor_parallelism_overlap_tiers()
        self.assertEqual(
            data["experiment_id"],
            "async_tensor_parallelism_gemm_comm_overlap_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier4_vs_tier1_layer_speedup_x"], 1.4)
        self.assertEqual(summary["tier4_exposed_comm_bubble_pct"], 1.6)
        self.assertEqual(summary["tier4_vs_tier2_gemm_retention_gain_pt"], 17.7)
        self.assertEqual(summary["tier4_vs_tier1_mfu_gain_pt"], 17.5)
        self.assertEqual(summary["tier4_vs_tier1_annual_savings_usd"], 750000)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "async_tensor_parallelism_gemm_comm_overlap_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_async_tensor_parallelism_overlap_tiers())


if __name__ == "__main__":
    unittest.main()
