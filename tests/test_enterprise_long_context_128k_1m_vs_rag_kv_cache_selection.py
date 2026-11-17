import unittest

from experiments.enterprise_long_context_128k_1m_vs_rag_kv_cache_selection.run import build_results


class TestEnterpriseLongContextVsRagSelection(unittest.TestCase):
    def test_long_context_and_hierarchical_rag_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 388)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_multihop_accuracy_pct"], 96.2)
        self.assertEqual(h["accuracy_gain_t4_vs_t1_pts"], 22.0)
        self.assertEqual(h["accuracy_gain_t4_vs_t2_pts"], 19.7)
        self.assertEqual(h["ttft_speedup_t4_vs_t1_x"], 21.1)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 91.7)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["multihop_accuracy_pct"], tiers[0]["multihop_accuracy_pct"])
        self.assertGreater(tiers[3]["max_concurrency_per_80g_gpu"], tiers[0]["max_concurrency_per_80g_gpu"])


if __name__ == "__main__":
    unittest.main()
