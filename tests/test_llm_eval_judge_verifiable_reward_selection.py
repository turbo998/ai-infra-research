from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.llm_eval_judge_verifiable_reward_selection.run import (
    evaluate_llm_eval_judge_tiers,
)


class TestLlmEvalJudgeVerifiableRewardSelection(unittest.TestCase):
    def test_provenance_and_eval_invariants(self) -> None:
        data = evaluate_llm_eval_judge_tiers()
        self.assertEqual(
            data["experiment_id"], "llm_eval_judge_verifiable_reward_selection"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier2_position_bias_flip_rate_pct"], 28.4)
        self.assertEqual(summary["tier3_position_bias_flip_rate_pct"], 2.1)
        self.assertEqual(summary["tier4_human_agreement_pct"], 96.8)
        self.assertEqual(summary["tier3_vs_tier2_cost_saving_pct"], 95.6)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "llm_eval_judge_verifiable_reward_selection"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_llm_eval_judge_tiers())


if __name__ == "__main__":
    unittest.main()
