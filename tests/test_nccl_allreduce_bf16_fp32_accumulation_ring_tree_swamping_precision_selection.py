import unittest
from experiments.nccl_allreduce_bf16_fp32_accumulation_ring_tree_swamping_precision_selection.run import evaluate_nccl_bf16_ring_tree_swamping


class TestNcclBf16RingTreeSwamping(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_nccl_bf16_ring_tree_swamping()
        self.assertEqual(res["round"], 324)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_hierarchical_two_stage_nvls_fp32_acc_plus_pairwise_tree_post_scale"]["small_gradient_signal_retained_pct"],
            99.0,
        )
        self.assertLess(
            tiers["tier4_hierarchical_two_stage_nvls_fp32_acc_plus_pairwise_tree_post_scale"]["allreduce_step_latency_ms"],
            50.0,
        )


if __name__ == "__main__":
    unittest.main()
