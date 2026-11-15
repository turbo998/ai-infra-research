import unittest
from experiments.gpu_nvlink_crc_flit_error_xid74_fabric_manager_drain_selection.run import (
    evaluate_nvlink_crc_flit_xid74_tiers,
)


class TestGpuNvlinkCrcFlitXid74Selection(unittest.TestCase):
    def test_nvlink_crc_xid74_tiers(self) -> None:
        data = evaluate_nvlink_crc_flit_xid74_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unmonitored_nvlink_crc_flit_replay_storm_straggler"]
        t2 = tiers["tier2_partial_sublink_drop_or_silent_pcie_shm_fallback"]
        t4 = tiers["tier4_proactive_crc_flit_slope_plus_144_sublink_prolog_gate"]
        self.assertFalse(t2["verifies_all_18_sublinks_active"])
        self.assertTrue(t4["verifies_all_18_sublinks_active"])
        self.assertEqual(t4["mid_run_xid74_job_crashes_per_month"], 0)
        self.assertGreater(t4["effective_tp8_allreduce_busbw_gbs"], t1["effective_tp8_allreduce_busbw_gbs"])


if __name__ == "__main__":
    unittest.main()
