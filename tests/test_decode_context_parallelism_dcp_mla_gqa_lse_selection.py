import unittest
from experiments.decode_context_parallelism_dcp_mla_gqa_lse_selection.run import (
    evaluate_dcp_mla_tiers,
)


class TestDecodeContextParallelismDcpSelection(unittest.TestCase):
    def test_dcp_mla_tiers(self) -> None:
        data = evaluate_dcp_mla_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_standard_tp8_replicated_mla_kv_cache"]
        t4 = tiers["tier4_hybrid_tp_weights_dcp_kv_fused_nvlink_lse_reduce"]
        self.assertEqual(t1["kv_replication_factor_x"], 8)
        self.assertEqual(t4["kv_replication_factor_x"], 1)
        self.assertGreater(data["workload"]["tbt_speedup_x"], 6.5)


if __name__ == "__main__":
    unittest.main()
