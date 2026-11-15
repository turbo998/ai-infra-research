import unittest
from experiments.llm_kv_cache_fragmentation_vllm_paged_block_size_mla_selection.run import (
    evaluate_paged_kv_block_size_tiers,
)


class TestPagedKvBlockSizeSelection(unittest.TestCase):
    def test_paged_kv_block_size_tiers(self) -> None:
        data = evaluate_paged_kv_block_size_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_uncoalesced_page_size_1_scatter_gather_bottleneck"]
        t3 = tiers["tier3_oversized_block_size_256_internal_fragmentation_and_prefix_miss"]
        t4 = tiers["tier4_tma_aligned_block_size_64_flashmla_plus_batched_rdma"]
        self.assertGreater(t3["internal_fragmentation_waste_pct"], 15.0)
        self.assertLess(t4["internal_fragmentation_waste_pct"], 2.0)
        self.assertGreater(t4["effective_attention_hbm_gbs"], 3000.0)
        self.assertGreater(t4["rdma_kv_transfer_gbps"], t1["rdma_kv_transfer_gbbs"] if "rdma_kv_transfer_gbbs" in t1 else t1["rdma_kv_transfer_gbps"])


if __name__ == "__main__":
    unittest.main()
