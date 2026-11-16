import unittest
from experiments.pytorch_einsum_contraction_order_intermediate_oom_bmm_linear_selection.run import evaluate


class TestPyTorchEinsumContractionOrderIntermediateOOMBmmLinearSelection(unittest.TestCase):
    def test_evaluate_metrics(self) -> None:
        res = evaluate()
        t1 = res["tiers"]["tier1_suboptimal_left_to_right_einsum"]
        t2 = res["tiers"]["tier2_permuted_noncontiguous_einsum"]
        t3 = res["tiers"]["tier3_eager_unfused_dot_einsum"]
        t4 = res["tiers"]["tier4_explicit_associative_bmm_linear"]

        self.assertAlmostEqual(t1["intermediate_hbm_gb"], 32.00, places=2)
        self.assertAlmostEqual(t1["latency_ms"], 18.40, places=2)
        self.assertAlmostEqual(t2["latency_ms"], 2.85, places=2)
        self.assertAlmostEqual(t3["latency_ms"], 1.95, places=2)
        self.assertAlmostEqual(t4["intermediate_hbm_gb"], 0.008, places=3)
        self.assertAlmostEqual(t4["latency_ms"], 0.12, places=2)
        self.assertAlmostEqual(t4["speedup_vs_tier1_x"], 153.3, places=1)
        self.assertAlmostEqual(t4["saved_hbm_gb"], 31.99, places=2)


if __name__ == "__main__":
    unittest.main()
