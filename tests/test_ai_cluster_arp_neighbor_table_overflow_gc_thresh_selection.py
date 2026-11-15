import unittest
from experiments.ai_cluster_arp_neighbor_table_overflow_gc_thresh_selection.run import (
    evaluate_arp_neighbor_tcpstore_storm_tiers,
)


class TestAiClusterArpNeighborTableOverflowSelection(unittest.TestCase):
    def test_arp_neighbor_tiers(self) -> None:
        data = evaluate_arp_neighbor_tcpstore_storm_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_default_linux_gc_thresh3_1024_arp_overflow_crash"]
        t4 = tiers["tier4_sysctl_plus_evpn_arp_suppression_and_hierarchical_rendezvous"]
        self.assertEqual(t1["arp_table_hard_limit"], 1024)
        self.assertGreater(t1["startup_timeout_failure_rate_pct"], 30.0)
        self.assertEqual(t4["startup_timeout_failure_rate_pct"], 0.0)
        self.assertLess(t4["init_process_group_latency_sec"], 5.0)


if __name__ == "__main__":
    unittest.main()
