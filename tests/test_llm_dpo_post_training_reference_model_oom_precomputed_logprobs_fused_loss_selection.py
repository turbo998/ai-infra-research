import unittest
from experiments.llm_dpo_post_training_reference_model_oom_precomputed_logprobs_fused_loss_selection.run import (
    evaluate_dpo_reference_tiers,
)


class TestDpoPostTrainingSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_dpo_reference_tiers()
        self.assertAlmostEqual(res["naive_peak_hbm_gb"], 88.40, places=2)
        self.assertAlmostEqual(res["tier4_peak_hbm_gb"], 51.20, places=2)
        self.assertAlmostEqual(res["hbm_saved_gb"], 37.20, places=2)
        self.assertAlmostEqual(res["throughput_speedup_x"], 2.45, places=2)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertFalse(t1["fits_80gb_h100"])
        self.assertTrue(t4["fits_80gb_h100"])
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
