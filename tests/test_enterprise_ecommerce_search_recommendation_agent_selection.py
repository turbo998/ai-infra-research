import unittest
from experiments.enterprise_ecommerce_search_recommendation_agent_selection.run import build_Payload


class TestEnterpriseEcommerceSearchRecommendationAgentSelection(unittest.TestCase):
    def test_tier4_maximizes_cvr_and_meets_sub100ms_sla(self) -> None:
        data = build_Payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_bm25_keyword_search"]
        t2 = tiers["tier2_sync_70b_llm_in_search_loop"]
        t4 = tiers["tier4_offline_llm_enrichment_dual_speed_retail_ai"]

        self.assertGreater(t4["long_tail_conversion_rate_pct"], t1["long_tail_conversion_rate_pct"] * 2.0)
        self.assertEqual(t4["price_stock_hallucination_pct"], 0.0)
        self.assertLess(t4["p95_search_grid_latency_ms"], 100.0)
        self.assertLess(t4["monthly_total_cost_usd"], t2["monthly_total_cost_usd"] * 0.1)

    def test_provenance_Explicit(self) -> None:
        data = build_Payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
