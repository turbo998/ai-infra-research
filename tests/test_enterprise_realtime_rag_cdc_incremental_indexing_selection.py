import unittest

from experiments.enterprise_realtime_rag_cdc_incremental_indexing_selection.run import build_results


class TestEnterpriseRealtimeRagCdcIncrementalSelection(unittest.TestCase):
    def test_cdc_freshness_and_ghost_chunk_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 390)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_p95_freshness_lag_sec"], 0.8)
        self.assertEqual(h["freshness_speedup_t4_vs_t1_x"], 63000.0)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 93.7)
        self.assertEqual(h["cost_savings_t4_vs_t2_pct"], 84.3)
        self.assertEqual(h["tier2_ghost_chunk_rate_pct"], 6.2)
        self.assertEqual(h["tier4_ghost_chunk_rate_pct"], 0.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["p95_freshness_lag_sec"], tiers[2]["p95_freshness_lag_sec"])
        self.assertEqual(tiers[3]["stale_ghost_chunk_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
