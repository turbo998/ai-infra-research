import unittest
from experiments.llm_context_parallel_ulysses_alltoall_vs_ring_hybrid_selection.run import (
    evaluate_context_parallel_ulysses_ring_usp_tiers,
)


class TestContextParallelUlyssesRingUspSelection(unittest.TestCase):
    def test_cp_usp_tiers(self) -> None:
        data = evaluate_context_parallel_ulysses_ring_usp_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pure_ulysses_cp64_exceeding_gqa_kv_heads_cross_node_alltoall"]
        t2 = tiers["tier2_pure_ring_attention_cp64_63_hops_and_small_chunk_launch_overhead"]
        t4 = tiers["tier4_2d_usp_intra_node_ulysses8_nvlink_plus_inter_node_zigzag_ring8_rdma"]
        self.assertEqual(t1["kv_head_replication_factor_x"], 8)
        self.assertEqual(t4["kv_head_replication_factor_x"], 1)
        self.assertLess(t4["layer_attention_latency_ms"], t2["layer_attention_latency_ms"] / 3.0)


if __name__ == "__main__":
    unittest.main()
