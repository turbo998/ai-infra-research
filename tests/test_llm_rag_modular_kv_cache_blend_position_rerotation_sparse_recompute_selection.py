import unittest
from experiments.llm_rag_modular_kv_cache_blend_position_rerotation_sparse_recompute_selection.run import evaluate_modular_kv_cache_blend


class TestModularKvCacheBlend(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_modular_kv_cache_blend()
        self.assertEqual(res["round"], 304)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_cacheblend_prerope_rerotation_plus_15pct_selective_cross_attn_recompute"]["ttft_latency_ms"],
            450.0,
        )
        self.assertGreater(
            tiers["tier4_cacheblend_prerope_rerotation_plus_15pct_selective_cross_attn_recompute"]["multihop_rag_f1_score"],
            77.5,
        )


if __name__ == "__main__":
    unittest.main()
