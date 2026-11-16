import unittest
from experiments.pytorch_inference_mode_vs_no_grad_version_counter_saved_for_backward_selection.run import evaluate


class TestPyTorchInferenceModeVsNoGradVersionCounterSavedForBackwardSelection(unittest.TestCase):
    def test_evaluate_metrics(self) -> None:
        res = evaluate()
        t1 = res["tiers"]["tier1_no_grad_everywhere"]
        t2 = res["tiers"]["tier2_raw_inference_mode_into_linear"]
        t3 = res["tiers"]["tier3_exit_inference_mode_without_clone"]
        t4 = res["tiers"]["tier4_inference_mode_with_boundary_clone"]

        self.assertAlmostEqual(t1["cpu_dispatch_latency_ms"], 4.85, places=2)
        self.assertEqual(t1["version_counter_allocs"], 960)
        self.assertAlmostEqual(t2["saved_for_backward_crash_pct"], 100.0, places=1)
        self.assertAlmostEqual(t3["saved_for_backward_crash_pct"], 100.0, places=1)
        self.assertAlmostEqual(t4["cpu_dispatch_latency_ms"], 2.92, places=2)
        self.assertAlmostEqual(t4["saved_for_backward_crash_pct"], 0.0, places=1)
        self.assertAlmostEqual(t4["speedup_vs_tier1_x"], 1.66, places=2)


if __name__ == "__main__":
    unittest.main()
