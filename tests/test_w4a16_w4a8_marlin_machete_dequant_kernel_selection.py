import unittest
from experiments.w4a16_w4a8_marlin_machete_dequant_kernel_selection.run import (
    evaluate_w4_dequant_kernel_tiers,
)


class TestMarlinMacheteKernelSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        data = evaluate_w4_dequant_kernel_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_two_step_dequant_to_hbm_cublas"]
        t2 = tiers["tier2_basic_awq_gptq_fused_int2float_kernel"]
        t4 = tiers["tier4_machete_hopper_wgmma_tma_w4a16_w4a8"]
        self.assertLess(t1["speedup_vs_fp16_batch1"], 1.0)
        self.assertLess(t2["speedup_vs_fp16_batch128"], 1.0)
        self.assertGreaterEqual(t4["speedup_vs_fp16_batch1"], 3.9)
        self.assertGreaterEqual(t4["speedup_vs_fp16_batch128"], 2.0)


if __name__ == "__main__":
    unittest.main()
