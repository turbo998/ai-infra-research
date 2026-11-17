import unittest

from experiments.enterprise_ai_finops_token_cost_governance_cascade_selection.run import build_results


class TestEnterpriseAiFinopsCascadeSelection(unittest.TestCase):
    def test_finops_cascade_and_soft_downgrade_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 398)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_quality_retention_pct"], 98.8)
        self.assertEqual(h["tier4_service_availability_pct"], 99.99)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 80.5)
        self.assertEqual(h["annual_dollar_savings_usd"], 1082400)
        self.assertEqual(h["availability_gain_t4_vs_t2_pts"], 10.99)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["monthly_spend_usd"], tiers[2]["monthly_spend_usd"])
        self.assertGreater(tiers[3]["service_availability_pct"], tiers[1]["service_availability_pct"])


if __name__ == "__main__":
    unittest.main()
