"""Unit tests for Round 76 LLM RL Post-Training (verl 3D-HybridEngine vs Slime Async Partial Rollout + TIS) Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.llm_rl_post_training_verl_slime_arch_selection.run import (
    evaluate_llm_rl_post_training_verl_slime_arch_matrix,
)


class TestLlmRlPostTrainingVerlSlimeArchSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_llm_rl_post_training_verl_slime_arch_matrix()
        self.assertEqual(
            data["experiment_id"],
            "llm_rl_post_training_verl_slime_arch_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier4_slime_areal_async_streaming_partial_rollout_tis"]["pipeline_and_straggler_bubble_pct"],
            6.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_throughput_gain_x"],
            2.5,
        )
        self.assertGreater(
            data["summary"]["tier3_vs_tier2_reshard_speedup_x"],
            15.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "llm_rl_post_training_verl_slime_arch_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_llm_rl_post_training_verl_slime_arch_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
