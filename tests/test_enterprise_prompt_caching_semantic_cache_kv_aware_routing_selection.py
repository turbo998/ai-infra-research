import unittest

from experiments.enterprise_prompt_caching_semantic_cache_kv_aware_routing_selection.run import build_results


class TestEnterprisePromptCachingKvAwareRoutingSelection(unittest.TestCase):
    def test_kv_aware_router_and_cache_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 379)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_effective_cache_hit_rate_pct"], 89.4)
        self.assertEqual(h["hit_rate_gain_t4_vs_t1_pts"], 70.9)
        self.assertEqual(h["ttft_speedup_t4_vs_t1_x"], 6.76)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 72.9)
        self.assertEqual(h["tier2_false_positive_rate_pct"], 4.8)
        self.assertEqual(h["tier4_false_positive_rate_pct"], 0.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["p95_ttft_ms"], tiers[2]["p95_ttft_ms"])
        self.assertEqual(tiers[3]["semantic_false_positive_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
