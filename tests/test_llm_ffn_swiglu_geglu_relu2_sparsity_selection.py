import unittest
from experiments.llm_ffn_swiglu_geglu_relu2_sparsity_selection.run import (
    evaluate_ffn_architecture_tiers,
)


class TestFfnSwigluRelu2Selection(unittest.TestCase):
    def test_tiers_and_sparsity(self) -> None:
        data = evaluate_ffn_architecture_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(data["workload"]["d_ff_swiglu"], 28672)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unfused_swiglu_3_separate_gemms"]
        t4 = tiers["tier4_squared_relu_2matrix_90pct_natural_sparsity"]
        self.assertEqual(t1["activation_zero_sparsity_pct"], 0.0)
        self.assertGreaterEqual(t4["activation_zero_sparsity_pct"], 90.0)
        self.assertLess(t4["decode_ffn_latency_ms"], 2.2)


if __name__ == "__main__":
    unittest.main()
