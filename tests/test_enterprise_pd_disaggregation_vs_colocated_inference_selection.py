import unittest

from experiments.enterprise_pd_disaggregation_vs_colocated_inference_selection.run import build_results


class TestEnterprisePdDisaggregationSelection(unittest.TestCase):
    def test_pd_disaggregation_goodput_and_sla_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 383)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_sla_goodput_tok_per_s"], 5420)
        self.assertEqual(h["goodput_gain_t4_vs_t2_x"], 2.19)
        self.assertEqual(h["goodput_gain_t4_vs_t1_x"], 4.71)
        self.assertEqual(h["cost_savings_t4_vs_t2_pct"], 45.7)
        self.assertEqual(h["tier4_p99_tpot_ms"], 21.5)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["sla_goodput_tok_per_s"], tiers[1]["sla_goodput_tok_per_s"])
        self.assertLess(tiers[3]["p99_tpot_ms"], 35.0)


if __name__ == "__main__":
    unittest.main()
