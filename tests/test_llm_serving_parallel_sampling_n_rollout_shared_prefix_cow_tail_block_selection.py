import unittest
from experiments.llm_serving_parallel_sampling_n_rollout_shared_prefix_cow_tail_block_selection.run import (
    evaluate_parallel_sampling_cow_tiers,
)


class TestParallelSamplingCowSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_parallel_sampling_cow_tiers()
        self.assertEqual(res["full_blocks"], 250)
        self.assertEqual(res["tail_rem_tokens"], 5)
        self.assertAlmostEqual(res["ttft_speedup_x"], 15.6, places=1)
        self.assertAlmostEqual(res["hbm_reduction_x"], 15.8, places=1)
        self.assertAlmostEqual(res["unprotected_tail_corruption_pct"], 93.75, places=2)
        t2 = res["tiers"][1]
        t4 = res["tiers"][3]
        self.assertEqual(t2["tail_block_kv_corruption_pct"], 93.75)
        self.assertEqual(t4["tail_block_kv_corruption_pct"], 0.0)
        self.assertEqual(t4["prompt_kv_hbm_gb"], 1.98)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
