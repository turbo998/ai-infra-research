import unittest

from experiments.enterprise_llm_evaluation_llmops_ci_cd_shadow_release_gate_selection.run import build_results


class TestEnterpriseLlmEvaluationShadowGateSelection(unittest.TestCase):
    def test_eval_and_shadow_gate_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 380)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_regression_catch_rate_pct"], 98.6)
        self.assertEqual(h["catch_rate_gain_t4_vs_t1_pts"], 60.6)
        self.assertEqual(h["tier4_human_kappa_agreement"], 0.91)
        self.assertEqual(h["kappa_gain_t4_vs_t2"], 0.35)
        self.assertEqual(h["cost_savings_t4_vs_t2_pct"], 77.0)
        self.assertEqual(h["release_speedup_t4_vs_t1_x"], 400.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["pre_prod_regression_catch_rate_pct"], tiers[2]["pre_prod_regression_catch_rate_pct"])
        self.assertGreater(tiers[3]["human_expert_kappa_agreement"], tiers[1]["human_expert_kappa_agreement"])


if __name__ == "__main__":
    unittest.main()
