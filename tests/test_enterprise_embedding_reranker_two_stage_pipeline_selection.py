import unittest

from experiments.enterprise_embedding_reranker_two_stage_pipeline_selection.run import build_results


class TestEnterpriseEmbeddingRerankerPipelineSelection(unittest.TestCase):
    def test_mrl_and_reranker_early_exit_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 400)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_ndcg_at_10_pct"], 92.4)
        self.assertEqual(h["ndcg_gain_t4_vs_t1_pts"], 15.6)
        self.assertEqual(h["latency_speedup_t4_vs_t2_x"], 20.0)
        self.assertEqual(h["latency_speedup_t4_vs_t3_x"], 3.96)
        self.assertEqual(h["cost_savings_t4_vs_t2_pct"], 91.6)
        self.assertEqual(h["tier4_p95_latency_ms"], 24.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["ndcg_at_10_pct"], tiers[2]["ndcg_at_10_pct"])
        self.assertLess(tiers[3]["p95_retrieval_latency_ms"], tiers[2]["p95_retrieval_latency_ms"])


if __name__ == "__main__":
    unittest.main()
