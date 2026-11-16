import unittest
from experiments.pytorch_amp_gradscaler_bf16_found_inf_sync_stall_fp8_scaling_selection.run import (
    evaluate_gradscaler_fp8_scaling_tiers,
)


class TestGradScalerFp8ScalingSelection(unittest.TestCase):
    def test_tiers_and_properties(self) -> None:
        res = evaluate_gradscaler_fp8_scaling_tiers()
        self.assertAlmostEqual(res["bf16_gradscaler_stall_ms"], 14.60, places=2)
        self.assertEqual(res["bf16_clean_stall_ms"], 0.00)
        self.assertAlmostEqual(res["unscale_hbm_saved_gb"], 28.0, places=1)
        self.assertAlmostEqual(res["fp8_blockwise_rel_throughput_pct"], 154.2, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["step_sync_unscale_overhead_ms"], 14.60)
        self.assertEqual(t4["step_sync_unscale_overhead_ms"], 0.00)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
