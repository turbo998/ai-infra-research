import unittest
from experiments.gpu_register_spilling_setmaxnreg_occupancy_selection.run import (
    evaluate_register_setmaxnreg_tiers,
)


class TestGpuRegisterSpillingSetmaxnregSelection(unittest.TestCase):
    def test_setmaxnreg_budget_and_tflops(self) -> None:
        data = evaluate_register_setmaxnreg_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertLessEqual(data["workload"]["total_used_regs_tier4"], 65536)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unconstrained_large_tile_register_spill"]
        t4 = tiers["tier4_hopper_setmaxnreg_asymmetric_warpgroup_reallocation"]
        self.assertGreater(t1["spilled_bytes_per_thread"], 0)
        self.assertEqual(t4["spilled_bytes_per_thread"], 0)
        self.assertGreaterEqual(t4["sustained_tflops"], 780.0)


if __name__ == "__main__":
    unittest.main()
