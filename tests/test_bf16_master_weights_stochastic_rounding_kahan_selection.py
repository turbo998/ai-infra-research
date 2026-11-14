import unittest
from experiments.bf16_master_weights_stochastic_rounding_kahan_selection.run import (
    evaluate_bf16_master_weight_tiers,
)


class TestBf16MasterWeightsSrKahanSelection(unittest.TestCase):
    def test_bf16_master_tiers(self) -> None:
        data = evaluate_bf16_master_weight_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_fp32_master_weights_baseline"]
        t2 = tiers["tier2_naive_bf16_master_weights_rne"]
        t4 = tiers["tier4_kahan_compensated_bf16_plus_sr_adamw"]
        self.assertGreater(t2["swallowed_updates_pct"], 80.0)
        self.assertEqual(t4["final_validation_loss"], t1["final_validation_loss"])
        self.assertLess(t4["total_weight_opt_hbm_gb"], t1["total_weight_opt_hbm_gb"])


if __name__ == "__main__":
    unittest.main()
