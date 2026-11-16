import unittest
from experiments.pytorch_adamw_foreach_temp_buffer_oom_fused_capturable_selection.run import evaluate


class TestPyTorchAdamWForeachTempBufferOOMFusedCapturableSelection(unittest.TestCase):
    def test_evaluate_metrics(self) -> None:
        res = evaluate()
        t1 = res["tiers"]["tier1_single_tensor_loop"]
        t2 = res["tiers"]["tier2_foreach_multi_tensor"]
        t3 = res["tiers"]["tier3_fused_float_lr_uncapturable"]
        t4 = res["tiers"]["tier4_fused_capturable_tensor_lr"]

        self.assertEqual(t1["kernel_launches"], 4350)
        self.assertAlmostEqual(t1["step_latency_ms"], 28.40, places=2)
        self.assertAlmostEqual(t2["temp_buffer_spike_gb"], 17.50, places=2)
        self.assertEqual(t3["recompilations_per_1k_steps"], 64)
        self.assertAlmostEqual(t4["temp_buffer_spike_gb"], 0.00, places=2)
        self.assertAlmostEqual(t4["step_latency_ms"], 3.80, places=2)
        self.assertAlmostEqual(t4["speedup_vs_tier1_x"], 7.5, places=1)
        self.assertAlmostEqual(t4["saved_temp_hbm_gb"], 17.50, places=2)


if __name__ == "__main__":
    unittest.main()
