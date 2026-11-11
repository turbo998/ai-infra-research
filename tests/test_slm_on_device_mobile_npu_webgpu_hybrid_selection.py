"""Unit tests for Round 63 Small Language Model (SLM) On-Device Mobile NPU, WebGPU & Cloud-Edge Hybrid Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.slm_on_device_mobile_npu_webgpu_hybrid_selection.run import (
    evaluate_slm_on_device_npu_hybrid_matrix,
)


class TestSlmOnDeviceMobileNpuWebgpuHybridSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_slm_on_device_npu_hybrid_matrix()
        self.assertEqual(
            data["experiment_id"],
            "slm_on_device_mobile_npu_webgpu_hybrid_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_confidence_gated_cloud_edge_hybrid_cascade"]["blended_task_accuracy_pct"],
            88.0,
        )
        self.assertLess(
            tiers["tier3_pure_on_device_npu_spinquant_w4a8"]["device_power_draw_watts"],
            3.0,
        )
        self.assertGreater(
            data["summary"]["tier3_vs_tier2_npu_energy_efficiency_ratio"],
            10.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "slm_on_device_mobile_npu_webgpu_hybrid_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_slm_on_device_npu_hybrid_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
