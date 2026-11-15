import unittest
from experiments.gpu_clock_throttling_sw_power_cap_hw_thermal_vr_droop_selection.run import (
    evaluate_gpu_clock_throttle_bitmask_tiers,
)


class TestGpuClockThrottlingBitmaskSelection(unittest.TestCase):
    def test_clock_throttle_bitmask_tiers(self) -> None:
        data = evaluate_gpu_clock_throttle_bitmask_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unmonitored_hw_power_brake_0x80_510mhz_straggler"]
        t2 = tiers["tier2_naive_nonzero_throttle_alert_sw_power_cap_0x04_storm"]
        t4 = tiers["tier4_bitmask_0xe8_hardware_slowdown_gate_plus_auto_cordon"]
        self.assertEqual(t1["throttled_gpu_sm_clock_mhz"], 510)
        self.assertGreater(t2["false_positive_alert_rate_pct"], 90.0)
        self.assertEqual(t4["false_positive_alert_rate_pct"], 0.0)
        self.assertGreater(t4["cluster_1024gpu_effective_mfu_pct"], 50.0)


if __name__ == "__main__":
    unittest.main()
