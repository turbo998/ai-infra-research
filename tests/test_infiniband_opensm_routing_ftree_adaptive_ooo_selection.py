import unittest
from experiments.infiniband_opensm_routing_ftree_adaptive_ooo_selection.run import (
    evaluate_infiniband_opensm_routing_tiers,
)


class TestInfiniBandOpenSmRoutingSelection(unittest.TestCase):
    def test_opensm_routing_tiers(self) -> None:
        data = evaluate_infiniband_opensm_routing_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_default_minhop_static_lft_credit_loop"]
        t4 = tiers["tier4_ar_ftree_ar_updn_plus_rnic_ooo_and_fast_recovery"]
        self.assertFalse(t1["credit_loop_deadlock_free"])
        self.assertTrue(t4["credit_loop_deadlock_free"])
        self.assertLess(t4["link_failover_latency_ms"], 1.0)
        self.assertGreaterEqual(t4["alltoall_busbw_gbs"], 49.0)


if __name__ == "__main__":
    unittest.main()
