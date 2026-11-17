import unittest

from experiments.enterprise_vector_database_billion_scale_hybrid_search_selection.run import build_results


class TestEnterpriseVectorDatabaseHybridSearchSelection(unittest.TestCase):
    def test_vector_tiered_storage_and_quantization_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 382)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_ram_required_gb"], 48)
        self.assertEqual(h["ram_reduction_t4_vs_t1_x"], 19.2)
        self.assertEqual(h["tier4_recall_at_10_pct"], 96.5)
        self.assertEqual(h["tier4_p95_latency_ms"], 19.0)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 91.5)
        self.assertEqual(h["cost_savings_t4_vs_t2_pct"], 82.2)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["monthly_infra_cost_usd"], tiers[2]["monthly_infra_cost_usd"])
        self.assertLess(tiers[3]["cluster_ram_gb"], tiers[0]["cluster_ram_gb"])


if __name__ == "__main__":
    unittest.main()
