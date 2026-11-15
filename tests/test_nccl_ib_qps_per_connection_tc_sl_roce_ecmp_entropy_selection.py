import unittest
from experiments.nccl_ib_qps_per_connection_tc_sl_roce_ecmp_entropy_selection.run import (
    evaluate_nccl_qps_per_connection_icm_tiers,
)


class TestNcclQpsPerConnectionIcmSelection(unittest.TestCase):
    def test_nccl_qps_icm_tiers(self) -> None:
        data = evaluate_nccl_qps_per_connection_icm_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_single_qp_elephant_flow_ecmp_hash_collision"]
        t3 = tiers["tier3_overprovisioned_qps_32_rnic_qpc_icm_cache_thrashing"]
        t4 = tiers["tier4_golden_qps_4_plus_split_data_1_within_icm_budget"]
        self.assertGreater(t3["rnic_qpc_icm_cache_miss_rate_pct"], 60.0)
        self.assertLess(t4["rnic_qpc_icm_cache_miss_rate_pct"], 1.0)
        self.assertGreater(t4["effective_busbw_gbps"], t1["effective_busbw_gbps"])
        self.assertGreater(t4["effective_busbw_gbps"], 360.0)


if __name__ == "__main__":
    unittest.main()
