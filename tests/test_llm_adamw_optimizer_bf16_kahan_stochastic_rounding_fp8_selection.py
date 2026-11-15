import unittest
from experiments.llm_adamw_optimizer_bf16_kahan_stochastic_rounding_fp8_selection.run import (
    evaluate_adamw_low_precision_tiers,
)


class TestAdamwLowPrecisionSelection(unittest.TestCase):
    def test_adamw_tiers(self) -> None:
        data = evaluate_adamw_low_precision_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pure_bf16_weights_and_moments_rne_stagnation"]
        t2 = tiers["tier2_standard_fp32_master_weights_and_fp32_m_v"]
        t4 = tiers["tier4_kahan_compensated_bf16_or_fp32_master_plus_block_fp8_moments"]
        self.assertGreater(t1["val_loss_gap_vs_fp32_master"], 0.3)
        self.assertEqual(t4["total_state_hbm_70b_gb"], 490.0)
        self.assertLess(t4["total_state_hbm_70b_gb"], t2["total_state_hbm_70b_gb"])
        self.assertLessEqual(t4["val_loss_gap_vs_fp32_master"], 0.002)


if __name__ == "__main__":
    unittest.main()
