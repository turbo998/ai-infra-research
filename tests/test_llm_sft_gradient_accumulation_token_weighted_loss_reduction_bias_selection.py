import unittest
from experiments.llm_sft_gradient_accumulation_token_weighted_loss_reduction_bias_selection.run import evaluate_grad_accum_loss_reduction_bias


class TestGradAccumLossReductionBias(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_grad_accum_loss_reduction_bias()
        self.assertEqual(res["round"], 326)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertEqual(
            tiers["tier4_global_valid_token_sum_allreduce_denominator_exact_equivalence"]["short_vs_long_token_weight_ratio_x"],
            1.0,
        )
        self.assertEqual(
            tiers["tier4_global_valid_token_sum_allreduce_denominator_exact_equivalence"]["gradient_deviation_vs_true_global_batch_pct"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
