import unittest
from experiments.gpu_pcie_aer_correctable_replay_link_retrain_xid79_selection.run import (
    evaluate_pcie_aer_replay_xid79_tiers,
)


class TestGpuPcieAerReplayXid79Selection(unittest.TestCase):
    def test_pcie_aer_xid79_tiers(self) -> None:
        data = evaluate_pcie_aer_replay_xid79_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unmonitored_aer_replay_storm_and_syslog_flood"]
        t2 = tiers["tier2_ltssm_silent_link_speed_width_downgrade_straggler"]
        t4 = tiers["tier4_proactive_dcgm_replay_slope_plus_lnksta_gen5x16_gate"]
        self.assertFalse(t2["silent_link_downgrade_caught"])
        self.assertTrue(t4["silent_link_downgrade_caught"])
        self.assertEqual(t4["unplanned_xid79_job_crashes_per_month"], 0)
        self.assertGreater(t4["effective_pcie_bandwidth_gbs"], t1["effective_pcie_bandwidth_gbs"])


if __name__ == "__main__":
    unittest.main()
