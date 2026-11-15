import unittest
from experiments.llm_long_context_sliding_window_attention_swa_sink_hybrid_kv_selection.run import evaluate_swa_sink_hybrid_kv


class TestSwaSinkHybridKv(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_swa_sink_hybrid_kv()
        self.assertEqual(res["round"], 293)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier3_5to1_hybrid_swa_global_with_naive_uniform_kv_allocator"]["kv_cache_128k_gb"],
            tiers["tier4_5to1_hybrid_split_circular_swa_ring_plus_global_kv_pool"]["kv_cache_128k_gb"] * 5.0,
        )
        self.assertGreater(
            tiers["tier4_5to1_hybrid_split_circular_swa_ring_plus_global_kv_pool"]["niah_128k_recall_pct"],
            99.0,
        )


if __name__ == "__main__":
    unittest.main()
