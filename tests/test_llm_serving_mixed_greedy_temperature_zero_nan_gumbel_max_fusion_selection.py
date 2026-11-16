import unittest
from experiments.llm_serving_mixed_greedy_temperature_zero_nan_gumbel_max_fusion_selection.run import evaluate_mixed_greedy_temperature_gumbel


class TestMixedGreedyTemperatureGumbel(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_mixed_greedy_temperature_gumbel()
        self.assertEqual(res["round"], 320)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_single_pass_multiplicative_gumbel_max_argmax_triton_kernel"]["sampler_latency_ms"],
            0.5,
        )
        self.assertEqual(
            tiers["tier4_single_pass_multiplicative_gumbel_max_argmax_triton_kernel"]["fp16_bf16_nan_crash_rate_pct"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
