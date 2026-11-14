import unittest
from experiments.gpu_hbm_ecc_row_remapping_xid63_xid64_selection.run import (
    evaluate_ecc_row_remap_tiers,
)


class TestGpuHbmEccRowRemappingSelection(unittest.TestCase):
    def test_ecc_remap_tiers(self) -> None:
        data = evaluate_ecc_row_remap_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_ignore_pending_remap_reschedule_without_reset"]
        t4 = tiers["tier4_predictive_sbe_bank_velocity_plus_xid63_reset_xid64_rma"]
        self.assertGreater(t1["repeat_dbe_crash_rate_pct"], 90.0)
        self.assertEqual(t4["repeat_dbe_crash_rate_pct"], 0.0)
        self.assertLess(t4["mttr_minutes"], 2.0)


if __name__ == "__main__":
    unittest.main()
