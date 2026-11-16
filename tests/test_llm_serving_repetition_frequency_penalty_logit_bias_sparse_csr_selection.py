import unittest
from experiments.llm_serving_repetition_frequency_penalty_logit_bias_sparse_csr_selection.run import (
    evaluate_penalty_kernel_tiers,
)


class TestPenaltySparseCsrSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_penalty_kernel_tiers()
        self.assertAlmostEqual(res["naive_32k_latency_ms"], 9.40, places=2)
        self.assertAlmostEqual(res["sparse_csr_32k_latency_ms"], 0.14, places=2)
        self.assertAlmostEqual(res["speedup_32k_x"], 67.1, places=1)
        self.assertAlmostEqual(res["workspace_reduction_x"], 61.2, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["penalty_latency_32k_ms"], 9.40)
        self.assertEqual(t4["penalty_latency_32k_ms"], 0.14)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
