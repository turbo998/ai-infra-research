import unittest
from experiments.llm_gradient_clipping_dtensor_async_norm_nan_inf_guard_selection.run import (
    evaluate_gradient_clipping_dtensor_sync_tiers,
)


class TestGradientClippingDTensorSyncSelection(unittest.TestCase):
    def test_gradient_clipping_tiers(self) -> None:
        data = evaluate_gradient_clipping_dtensor_sync_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_per_tensor_clip_and_tp_replicate_double_count"]
        t4 = tiers["tier4_dtensor_foreach_plus_device_clamp_and_capturable_fused_adamw"]
        self.assertGreater(t1["global_grad_norm_relative_error_pct"], 40.0)
        self.assertEqual(t4["global_grad_norm_relative_error_pct"], 0.0)
        self.assertEqual(t4["host_device_syncs_per_step"], 0)
        self.assertLess(t4["clip_and_step_overhead_ms"], 1.0)


if __name__ == "__main__":
    unittest.main()
