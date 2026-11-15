import unittest
from experiments.llm_gqa_mqa_tensor_parallel_kv_head_replication_dp_attention_selection.run import (
    evaluate_gqa_mla_tp_kv_replication_tiers,
)


class TestGqaMlaTpKvReplicationSelection(unittest.TestCase):
    def test_gqa_mla_tp_tiers(self) -> None:
        data = evaluate_gqa_mla_tp_kv_replication_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_tp8_exceeding_kv_heads_8x_kv_cache_replication"]
        t4 = tiers["tier4_decoupled_dp_attention_or_sequence_dcp_plus_tp_ep_ffn"]
        self.assertEqual(t1["kv_cache_replication_factor_x"], 8.0)
        self.assertEqual(t4["kv_cache_replication_factor_x"], 1.0)
        self.assertGreater(t4["max_concurrent_16k_requests"], 140)
        self.assertGreater(t4["serving_throughput_tok_per_sec"], 5000)


if __name__ == "__main__":
    unittest.main()
