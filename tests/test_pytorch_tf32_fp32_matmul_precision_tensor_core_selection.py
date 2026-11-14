import unittest
from experiments.pytorch_tf32_fp32_matmul_precision_tensor_core_selection.run import (
    evaluate_tf32_matmul_precision_tiers,
)


class TestPyTorchTf32MatmulPrecisionSelection(unittest.TestCase):
    def test_tf32_precision_tiers(self) -> None:
        data = evaluate_tf32_matmul_precision_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pytorch_default_highest_ieee_fp32_simt"]
        t3 = tiers["tier3_tf32_tensor_core_high_precision_mode"]
        self.assertLess(t1["sustained_fp32_op_tflops"], 50.0)
        self.assertGreater(t3["sustained_fp32_op_tflops"], 350.0)
        self.assertEqual(t3["accumulator_bits"], 32)


if __name__ == "__main__":
    unittest.main()
