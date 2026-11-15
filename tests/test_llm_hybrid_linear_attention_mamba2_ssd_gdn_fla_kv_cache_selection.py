import unittest
from experiments.llm_hybrid_linear_attention_mamba2_ssd_gdn_fla_kv_cache_selection.run import (
    evaluate_hybrid_linear_attention_serving_tiers,
)


class TestHybridLinearAttentionServingSelection(unittest.TestCase):
    def test_hybrid_linear_tiers(self) -> None:
        data = evaluate_hybrid_linear_attention_serving_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pure_linear_attention_mamba2_without_full_attention_layers"]
        t2 = tiers["tier2_pure_full_softmax_gqa_across_all_64_layers"]
        t4 = tiers["tier4_3to1_hybrid_gdn_gqa_plus_speculative_state_ring_and_interval_snapshots"]
        self.assertLess(t1["long_context_256k_ruler_pct"], 40.0)
        self.assertGreater(t4["long_context_256k_ruler_pct"], 92.0)
        self.assertLess(t4["per_req_256k_cache_hbm_gb"], t2["per_req_256k_cache_hbm_gb"] / 3.0)
        self.assertLess(t4["effective_decode_itl_ms"], 6.0)


if __name__ == "__main__":
    unittest.main()
