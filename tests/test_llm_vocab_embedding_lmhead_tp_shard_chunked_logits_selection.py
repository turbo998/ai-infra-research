import unittest
from experiments.llm_vocab_embedding_lmhead_tp_shard_chunked_logits_selection.run import (
    evaluate_lmhead_fused_linear_cross_entropy_tiers,
)


class TestLmHeadFusedLinearCrossEntropySelection(unittest.TestCase):
    def test_lmhead_cce_tiers(self) -> None:
        data = evaluate_lmhead_fused_linear_cross_entropy_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_full_logits_fp32_upcast_materialization_oom"]
        t4 = tiers["tier4_fused_linear_cross_entropy_liger_cce_sram_online_lse"]
        self.assertGreater(t1["peak_lmhead_hbm_gb"], 35.0)
        self.assertLess(t4["peak_lmhead_hbm_gb"], 0.2)
        self.assertFalse(t4["materializes_full_nv_logits_in_hbm"])
        self.assertLess(t4["lmhead_fwd_bwd_latency_ms"], 45.0)


if __name__ == "__main__":
    unittest.main()
