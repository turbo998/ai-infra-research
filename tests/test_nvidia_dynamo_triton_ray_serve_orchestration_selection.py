"""Unit tests for Enterprise Distributed Inference Orchestration (Triton vs. KServe vs. Ray Serve vs. NVIDIA Dynamo + NIXL) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.nvidia_dynamo_triton_ray_serve_orchestration_selection.run import (
    evaluate_nvidia_dynamo_triton_ray_serve_orchestration_matrix,
)


class TestNvidiaDynamoTritonRayServeOrchestrationSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_nvidia_dynamo_triton_ray_serve_orchestration_matrix()
        self.assertEqual(
            data["experiment_id"],
            "nvidia_dynamo_triton_ray_serve_orchestration_selection",
        )
        self.assertEqual(data["round"], 181)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_nvidia_dynamo_nixl_disaggregated_serving"]["sla_goodput_tok_per_gpu_s"],
            650.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_goodput_speedup_ratio"],
            3.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_cost_reduction_pct"],
            65.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "nvidia_dynamo_triton_ray_serve_orchestration_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_nvidia_dynamo_triton_ray_serve_orchestration_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
