import unittest
from experiments.gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection.run import (
    evaluate_hbm_row_remapping_tiers,
)


class TestHbmRowRemappingSelection(unittest.TestCase):
    def test_row_remapping_tiers(self) -> None:
        data = evaluate_hbm_row_remapping_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_reschedule_jobs_while_remapped_rows_pending_without_gpu_reset"]
        t4 = tiers["tier4_closed_loop_pending_reset_gate_plus_xid64_failure_rma_cordon"]
        self.assertGreater(t1["repeat_job_crash_rate_pct"], 80.0)
        self.assertEqual(t4["repeat_job_crash_rate_pct"], 0.0)
        self.assertEqual(t4["unnecessary_hardware_rma_rate_pct"], 0.0)
        self.assertEqual(t4["healing_downtime_sec"], 8.0)


if __name__ == "__main__":
    unittest.main()
