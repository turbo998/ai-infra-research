import unittest
from experiments.pytorch_meta_device_init_to_empty_uninitialized_nan_fsdp2_reset_parameters_selection.run import (
    evaluate_meta_device_init_tiers,
)


class TestMetaDeviceInitSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_meta_device_init_tiers()
        self.assertAlmostEqual(res["naive_host_ram_gb"], 1120.0, places=1)
        self.assertAlmostEqual(res["tier4_host_ram_gb"], 0.20, places=2)
        self.assertAlmostEqual(res["init_speedup_x"], 102.8, places=1)
        t1 = res["tiers"][0]
        t2 = res["tiers"][1]
        t4 = res["tiers"][3]
        self.assertEqual(t1["peak_node_ram_gb"], 1120.0)
        self.assertEqual(t2["step1_nan_rate_pct"], 100.0)
        self.assertEqual(t4["step1_nan_rate_pct"], 0.0)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
