import unittest
from experiments.cuda_device_side_assert_embedding_oob_sticky_context_destruction_guard_selection.run import (
    evaluate_cuda_assert_guard_tiers,
)


class TestCudaDeviceSideAssertSelection(unittest.TestCase):
    def test_tiers_and_survival(self) -> None:
        res = evaluate_cuda_assert_guard_tiers()
        self.assertEqual(res["naive_survival_pct"], 0.0)
        self.assertEqual(res["tier4_survival_pct"], 100.0)
        self.assertAlmostEqual(res["gpu_any_sync_stall_ms"], 1.85, places=2)
        self.assertAlmostEqual(res["tier4_overhead_ms"], 0.02, places=2)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["cucontext_survival_rate_pct"], 0.0)
        self.assertEqual(t4["cucontext_survival_rate_pct"], 100.0)
        self.assertTrue(t4["cudagraph_compatible"])
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
