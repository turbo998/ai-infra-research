import unittest
from experiments.long_context_parallelism_ulysses_ring_usp_selection.run import (
    evaluate_long_context_parallelism_tiers,
)


class TestLongContextParallelismSelection(unittest.TestCase):
    def test_tiers_and_usp_mfu(self) -> None:
        data = evaluate_long_context_parallelism_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_megatron_tp8_sp_naive_ring"] if "tier1_megatron_tp8_sp_naive_ring" in tiers else tiers["tier1_megatron_tp8_sp_naive_contiguous_ring"]
        t4 = tiers["tier4_hybrid_2d_usp_ulysses8_nvlink_plus_zigzag_ring4_rdma"]
        self.assertGreater(t1["causal_load_imbalance_ratio"], 5.0)
        self.assertEqual(t4["causal_load_imbalance_ratio"], 1.0)
        self.assertGreaterEqual(t4["cluster_32gpu_mfu_pct"], 54.0)


if __name__ == "__main__":
    unittest.main()
