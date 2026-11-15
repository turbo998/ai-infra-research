import unittest
from experiments.ai_cluster_roce_infiniband_pfc_deadlock_watchdog_ttl_selection.run import (
    evaluate_rocev2_pfc_watchdog_deadlock_tiers,
)


class TestRocev2PfcWatchdogDeadlockSelection(unittest.TestCase):
    def test_pfc_watchdog_tiers(self) -> None:
        data = evaluate_rocev2_pfc_watchdog_deadlock_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_no_pfc_watchdog_cascading_leaf_spine_pause_tree_freeze"]
        t4 = tiers["tier4_bpg_valley_free_plus_rnic_50ms_stall_guard_plus_switch_200ms_watchdog_drop"]
        self.assertEqual(t1["healthy_jobs_busbw_during_storm_gbps"], 0.0)
        self.assertEqual(t1["fabric_wide_blast_radius_pct"], 100.0)
        self.assertEqual(t4["fabric_wide_blast_radius_pct"], 0.0)
        self.assertGreater(t4["healthy_jobs_busbw_during_storm_gbps"], 380.0)


if __name__ == "__main__":
    unittest.main()
