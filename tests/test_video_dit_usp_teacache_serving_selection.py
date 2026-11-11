"""Unit tests for Round 44 Video DiT USP + TeaCache + SageAttention Serving Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.video_dit_usp_teacache_serving_selection.run import (
    evaluate_video_dit_usp_teacache_serving_matrix,
)


class TestVideoDitUspTeacacheServingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_video_dit_usp_teacache_serving_matrix()
        self.assertEqual(data["experiment_id"], "video_dit_usp_teacache_serving_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier3_xdit_usp_cfg_ulysses_hybrid"]["e2e_video_latency_s"],
            tiers["tier1_naive_tensor_parallel_tp8_allreduce"]["e2e_video_latency_s"] * 0.45,
        )
        self.assertLess(
            tiers["tier4_usp_teacache_sageattention_fp8_int8"]["e2e_video_latency_s"],
            7.0,
        )
        self.assertGreaterEqual(
            tiers["tier4_usp_teacache_sageattention_fp8_int8"]["vbench_quality_score_pct"],
            84.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_speedup_ratio"],
            8.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "video_dit_usp_teacache_serving_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_video_dit_usp_teacache_serving_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
