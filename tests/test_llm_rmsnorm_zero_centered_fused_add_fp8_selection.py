import unittest
from experiments.llm_rmsnorm_zero_centered_fused_add_fp8_selection.run import (
    evaluate_rmsnorm_fusion_tiers,
)


class TestRmsnormZeroCenteredFusedSelection(unittest.TestCase):
    def test_rmsnorm_tiers(self) -> None:
        data = evaluate_rmsnorm_fusion_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unfused_pytorch_eager_rmsnorm_residual"]
        t3 = tiers["tier3_fused_add_rmsnorm_in_place_residual_kernel"]
        t4 = tiers["tier4_single_pass_add_zero_centered_rmsnorm_fp8_quant"]
        self.assertGreater(t1["total_80layer_latency_us"], 200.0)
        self.assertEqual(t3["extra_activation_hbm_gb"], 0.0)
        self.assertLessEqual(t4["total_80layer_latency_us"], 45.0)


if __name__ == "__main__":
    unittest.main()
