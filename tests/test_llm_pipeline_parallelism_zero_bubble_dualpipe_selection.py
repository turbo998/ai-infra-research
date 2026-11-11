"""Unit tests for Round 60 LLM Pipeline Parallelism (1F1B vs Zero-Bubble ZB-V vs DeepSeek DualPipe) Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.llm_pipeline_parallelism_zero_bubble_dualpipe_selection.run import (
    evaluate_llm_pipeline_parallelism_dualpipe_matrix,
)


class TestLlmPipelineParallelismZeroBubbleDualpipeSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_llm_pipeline_parallelism_dualpipe_matrix()
        self.assertEqual(
            data["experiment_id"],
            "llm_pipeline_parallelism_zero_bubble_dualpipe_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier3_zero_bubble_zbv_bw_split"]["pipeline_bubble_pct"],
            3.0,
        )
        self.assertLess(
            tiers["tier4_deepseek_v3_dualpipe_bidirectional_overlap"]["exposed_comm_stall_pct"],
            1.5,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_total_stall_reduction_ratio"],
            10.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "llm_pipeline_parallelism_zero_bubble_dualpipe_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_llm_pipeline_parallelism_dualpipe_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
