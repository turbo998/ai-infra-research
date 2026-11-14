import unittest
from experiments.ai_cluster_ptp_ieee1588_clock_sync_trace_selection.run import (
    evaluate_ptp_clock_sync_tiers,
)


class TestPtpClockSyncSelection(unittest.TestCase):
    def test_tiers_and_skew(self) -> None:
        data = evaluate_ptp_clock_sync_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_standard_ntp_chrony_kernel_socket"]
        t4 = tiers["tier4_synce_plus_ptp_boundary_clock_gpu_timer_sync"]
        self.assertGreater(t1["max_clock_skew_us"], 1000.0)
        self.assertLessEqual(t4["max_clock_skew_ns"], 50)
        self.assertGreaterEqual(t4["straggler_attribution_accuracy_pct"], 99.5)


if __name__ == "__main__":
    unittest.main()
