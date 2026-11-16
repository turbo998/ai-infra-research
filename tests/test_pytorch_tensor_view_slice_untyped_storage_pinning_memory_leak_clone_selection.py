import unittest
from experiments.pytorch_tensor_view_slice_untyped_storage_pinning_memory_leak_clone_selection.run import (
    evaluate_view_storage_leak_tiers,
)


class TestTensorViewStorageLeakSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_view_storage_leak_tiers()
        self.assertEqual(res["storage_amplification_x"], 8192)
        self.assertAlmostEqual(res["uncloned_32steps_gb"], 32.00, places=2)
        self.assertAlmostEqual(res["cloned_32steps_gb"], 0.004, places=3)
        self.assertAlmostEqual(res["lmhead_speedup_x"], 510.0, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["retained_hbm_32steps_gb"], 32.00)
        self.assertEqual(t4["retained_hbm_32steps_gb"], 0.004)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
