"""Unit tests for Round 55 Embodied AI & Vision-Language-Action (VLA) Real-Time Inference Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.vla_embodied_robotics_realtime_inference_selection.run import (
    evaluate_vla_embodied_robotics_inference_matrix,
)


class TestVlaEmbodiedRoboticsRealtimeInferenceSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_vla_embodied_robotics_inference_matrix()
        self.assertEqual(data["experiment_id"], "vla_embodied_robotics_realtime_inference_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_dual_system_async_vlm_plus_1step_flow_trt"]["effective_control_freq_hz"],
            100.0,
        )
        self.assertGreater(
            tiers["tier4_dual_system_async_vlm_plus_1step_flow_trt"]["dexterous_task_success_pct"],
            tiers["tier1_autoregressive_discrete_action_tokens_openvla"]["dexterous_task_success_pct"] + 40.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_latency_speedup_ratio"],
            35.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "vla_embodied_robotics_realtime_inference_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_vla_embodied_robotics_inference_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
