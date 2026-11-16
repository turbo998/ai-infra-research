import unittest
from experiments.llm_rl_rollout_trainer_logprob_numerical_drift_is_ratio_spike_selection.run import evaluate_rl_rollout_trainer_logprob_drift


class TestRlRolloutTrainerLogprobDrift(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_rl_rollout_trainer_logprob_drift()
        self.assertEqual(res["round"], 310)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_trainer_old_logprob_reforward_plus_truncated_is_dual_clip_c3"]["step0_is_ratio_p99_deviation_pct"],
            1.0,
        )
        self.assertGreater(
            tiers["tier4_trainer_old_logprob_reforward_plus_truncated_is_dual_clip_c3"]["math_benchmark_pass_at_1_pct"],
            70.0,
        )


if __name__ == "__main__":
    unittest.main()
