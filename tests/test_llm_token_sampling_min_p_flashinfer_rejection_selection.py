import unittest
from experiments.llm_token_sampling_min_p_flashinfer_rejection_selection.run import (
    evaluate_token_sampling_tiers,
)


class TestTokenSamplingSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        data = evaluate_token_sampling_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertGreater(data["workload"]["kernel_speedup_x"], 20.0)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pytorch_eager_sort_cumsum_top_p"]
        t4 = tiers["tier4_min_p_dynamic_floor_plus_flashinfer_fused_kernel"]
        self.assertGreater(t1["sampling_kernel_latency_ms"], 1.5)
        self.assertLessEqual(t4["sampling_kernel_latency_ms"], 0.1)
        self.assertGreaterEqual(t4["high_temp_creative_coherence_score"], 90.0)


if __name__ == "__main__":
    unittest.main()
