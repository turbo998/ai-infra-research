import unittest
from experiments.nccl_pat_parallel_aggregated_trees_selection.run import (
    evaluate_nccl_pat_vs_ring_tiers,
)


class TestNcclPatParallelAggregatedTreesSelection(unittest.TestCase):
    def test_nccl_pat_tiers(self) -> None:
        data = evaluate_nccl_pat_vs_ring_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_flat_ring_1023_hops_latency_bound_trap"]
        t4 = tiers["tier4_nccl_pat_parallel_aggregated_binomial_trees_7_hops"]
        self.assertEqual(t1["inter_node_hop_steps"], 1023)
        self.assertEqual(t4["inter_node_hop_steps"], 7)
        self.assertLess(t4["collective_latency_us"], 1200.0)
        self.assertGreater(t4["effective_algbw_gbs"], 50.0)


if __name__ == "__main__":
    unittest.main()
