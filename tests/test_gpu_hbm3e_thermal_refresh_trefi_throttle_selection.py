import unittest
from experiments.gpu_hbm3e_thermal_refresh_trefi_throttle_selection.run import (
    evaluate_hbm_thermal_tiers,
)


class TestGpuHbmThermalSelection(unittest.TestCase):
    def test_tiers_and_bandwidth(self) -> None:
        data = evaluate_hbm_thermal_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_tim_pumpout_hbm_90c_4x_refresh_and_sw_throttle"]
        t4 = tiers["tier4_liquid_coldplate_below_72c_plus_delta_t_proactive_cordon"]
        self.assertGreater(t1["delta_t_mem_minus_gpu_c"], 20)
        self.assertLess(t1["effective_hbm_bw_tbs"], 3.5)
        self.assertEqual(t4["trefi_refresh_multiplier"], 1)
        self.assertGreaterEqual(t4["effective_hbm_bw_tbs"], 4.6)


if __name__ == "__main__":
    unittest.main()
