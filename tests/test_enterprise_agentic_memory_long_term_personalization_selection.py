import unittest

from experiments.enterprise_agentic_memory_long_term_personalization_selection.run import build_results


class TestEnterpriseAgenticMemorySelection(unittest.TestCase):
    def test_bitemporal_memory_and_contradiction_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 394)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_temporal_fact_accuracy_pct"], 96.8)
        self.assertEqual(h["accuracy_gain_t4_vs_t1_pts"], 42.8)
        self.assertEqual(h["accuracy_gain_t4_vs_t3_pts"], 22.3)
        self.assertEqual(h["ttft_speedup_t4_vs_t1_x"], 5.32)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 82.4)
        self.assertEqual(h["tier3_contradiction_rate_pct"], 26.5)
        self.assertEqual(h["tier4_contradiction_rate_pct"], 0.6)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["temporal_fact_accuracy_pct"], tiers[2]["temporal_fact_accuracy_pct"])
        self.assertLess(tiers[3]["stale_fact_contradiction_rate_pct"], tiers[2]["stale_fact_contradiction_rate_pct"])


if __name__ == "__main__":
    unittest.main()
