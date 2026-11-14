import unittest
from experiments.llm_training_loss_spike_qknorm_zloss_fp32_selection.run import (
    evaluate_training_stability_tiers,
)


class TestTrainingLossSpikeSelection(unittest.TestCase):
    def test_tiers_and_stability(self) -> None:
        data = evaluate_training_stability_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_standard_pre_ln_bf16_residual_no_qknorm"]
        t2 = tiers["tier2_attention_logit_tanh_soft_capping"]
        t4 = tiers["tier4_fused_qknorm_plus_fp32_residual_plus_zloss"]
        self.assertGreater(t1["max_attention_logit"], 100.0)
        self.assertGreater(t2["flashattention_kernel_overhead_pct"], 10.0)
        self.assertEqual(t4["nan_loss_spikes_per_10t_tokens"], 0)
        self.assertLessEqual(t4["flashattention_kernel_overhead_pct"], 1.0)


if __name__ == "__main__":
    unittest.main()
