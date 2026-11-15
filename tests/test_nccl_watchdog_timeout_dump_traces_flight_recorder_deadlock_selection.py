import unittest
from experiments.nccl_watchdog_timeout_dump_traces_flight_recorder_deadlock_selection.run import (
    evaluate_nccl_watchdog_flightrecorder_tiers,
)


class TestNcclWatchdogFlightRecorderSelection(unittest.TestCase):
    def test_nccl_watchdog_tiers(self) -> None:
        data = evaluate_nccl_watchdog_flightrecorder_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_default_30min_watchdog_victim_log_misattribution"]
        t4 = tiers["tier4_flight_recorder_plus_nccl_ras_and_seqnum_set_diff_analyzer"]
        self.assertEqual(t1["patient_zero_attribution_accuracy_pct"], 0.0)
        self.assertEqual(t4["patient_zero_attribution_accuracy_pct"], 100.0)
        self.assertTrue(t4["distinguishes_cpu_vs_network_hang"])
        self.assertLess(t4["hang_waste_duration_sec"], 20.0)


if __name__ == "__main__":
    unittest.main()
