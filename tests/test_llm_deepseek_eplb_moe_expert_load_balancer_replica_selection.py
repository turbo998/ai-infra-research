import unittest
from experiments.llm_deepseek_eplb_moe_expert_load_balancer_replica_selection.run import (
    evaluate_moe_eplb_load_balancing_tiers,
)


class TestMoeEplbLoadBalancingSelection(unittest.TestCase):
    def test_moe_eplb_tiers(self) -> None:
        data = evaluate_moe_eplb_load_balancing_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_static_1to1_expert_placement_256_slots"]
        t4 = tiers["tier4_deepseek_eplb_hierarchical_node_packing_plus_hot_replicas_and_live_swap"]
        self.assertGreater(t1["max_to_mean_gpu_load_ratio_x"], 4.5)
        self.assertLess(t4["max_to_mean_gpu_load_ratio_x"], 1.15)
        self.assertLess(t4["moe_layer_latency_ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
