import unittest
from experiments.moe_dp_attention_ep_expert_parallelism_selection.run import (
    evaluate_dp_attention_ep_moe_tiers,
)


class TestMoeDpAttentionEpSelection(unittest.TestCase):
    def test_dp_attention_tiers(self) -> None:
        data = evaluate_dp_attention_ep_moe_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pure_tp_attention_plus_tp_moe"]
        t4 = tiers["tier4_token_balanced_dp_attention_ep64_cudagraph_tbo"]
        self.assertEqual(t1["kv_cache_duplication_x"], 8)
        self.assertEqual(t4["kv_cache_duplication_x"], 1)
        self.assertEqual(t4["attention_allreduce_per_layer"], 0)
        self.assertGreater(data["workload"]["throughput_gain_x"], 3.0)


if __name__ == "__main__":
    unittest.main()
