import unittest

from experiments.enterprise_slm_distillation_synthetic_data_flywheel_selection.run import build_results


class TestEnterpriseSlmDistillationFlywheelSelection(unittest.TestCase):
    def test_distillation_parity_and_flywheel_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 392)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_teacher_parity_pct"], 99.4)
        self.assertEqual(h["parity_gain_t4_vs_t1_pts"], 20.9)
        self.assertEqual(h["parity_gain_t4_vs_t2_pts"], 15.6)
        self.assertEqual(h["serving_savings_t4_vs_teacher_pct"], 81.6)
        self.assertEqual(h["build_cost_savings_t4_vs_t1_pct"], 90.2)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["teacher_parity_pct"], tiers[2]["teacher_parity_pct"])
        self.assertLess(tiers[3]["initial_build_cost_usd"], tiers[0]["initial_build_cost_usd"])


if __name__ == "__main__":
    unittest.main()
