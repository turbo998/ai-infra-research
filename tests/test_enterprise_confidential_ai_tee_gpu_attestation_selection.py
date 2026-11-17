import unittest

from experiments.enterprise_confidential_ai_tee_gpu_attestation_selection.run import build_results


class TestEnterpriseConfidentialAiTeeSelection(unittest.TestCase):
    def test_confidential_gpu_overhead_and_tco_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 386)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_throughput_overhead_pct"], 5.8)
        self.assertEqual(h["tier4_data_in_use_protection_pct"], 100.0)
        self.assertEqual(h["tco_savings_t4_vs_t2_pct"], 55.2)
        self.assertEqual(h["lead_time_speedup_t4_vs_t2_x"], 135.0)
        self.assertEqual(h["price_premium_t4_vs_t1_pct"], 15.8)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["monthly_tco_usd"], tiers[1]["monthly_tco_usd"])
        self.assertEqual(tiers[3]["data_in_use_protection_score_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
