import unittest
from experiments.llm_rl_kl_divergence_schulman_k3_estimator_vocab_oom_selection.run import (
    evaluate_rl_kl_estimator_tiers,
)


class TestRlKlEstimatorSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_rl_kl_estimator_tiers()
        self.assertAlmostEqual(res["full_vocab_kl_hbm_gb"], 39.86, places=2)
        self.assertAlmostEqual(res["hbm_saved_gb"], 39.86, places=2)
        self.assertAlmostEqual(res["k1_variance_ratio_x"], 14.8, places=1)
        self.assertEqual(res["clamped_k3_nan_pct"], 0.0)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["negative_kl_token_rate_pct"], 48.5)
        self.assertEqual(t4["negative_kl_token_rate_pct"], 0.0)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
