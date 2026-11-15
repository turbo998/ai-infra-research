import unittest
from experiments.llm_flashattention_deterministic_backward_splitkv_atomic_workspace_selection.run import (
    evaluate_flashattention_deterministic_bwd_tiers,
)


class TestFlashAttentionDeterministicBwdSelection(unittest.TestCase):
    def test_fa_deterministic_bwd_tiers(self) -> None:
        data = evaluate_flashattention_deterministic_bwd_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_default_fa2_fa3_non_deterministic_dq_atomicadd"]
        t4 = tiers["tier4_fa3_ordered_semaphore_or_sm_bounded_split_dq_deterministic"]
        self.assertFalse(t1["bitwise_reproducible"])
        self.assertTrue(t4["bitwise_reproducible"])
        self.assertEqual(t4["max_ulp_drift_per_step"], 0)
        self.assertGreater(t4["bwd_effective_tflops"], 750.0)


if __name__ == "__main__":
    unittest.main()
