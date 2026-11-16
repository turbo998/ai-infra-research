import unittest
from experiments.llm_linear_backward_wgrad_main_grad_accumulation_fusion_dgrad_first_selection.run import (
    evaluate_wgrad_fusion_tiers,
)


class TestWgradFusionSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_wgrad_fusion_tiers()
        self.assertAlmostEqual(res["extra_hbm_saved_gb"], 14.00, places=2)
        self.assertAlmostEqual(res["epilogue_speedup_x"], 15.5, places=1)
        self.assertAlmostEqual(res["naive_pp_bubble_pct"], 23.8, places=1)
        self.assertAlmostEqual(res["deferred_pp_bubble_pct"], 4.2, places=1)
        self.assertAlmostEqual(res["throughput_gain_pct"], 27.4, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["extra_grad_hbm_gb"], 14.00)
        self.assertEqual(t4["extra_grad_hbm_gb"], 0.00)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
