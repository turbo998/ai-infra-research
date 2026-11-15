import unittest
from experiments.llm_moe_alltoall_ep_node_limit_hierarchical_routing_selection.run import evaluate_moe_node_limited_routing


class TestMoeNodeLimitedRouting(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_moe_node_limited_routing()
        self.assertEqual(res["round"], 287)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_unconstrained_topk8_flat_alltoall_bf16"]["alltoall_latency_ms"],
            tiers["tier4_m4_2stage_deepep_plus_fp8_dispatch_bf16_combine_sm_free"]["alltoall_latency_ms"] * 4.5,
        )
        self.assertEqual(
            tiers["tier4_m4_2stage_deepep_plus_fp8_dispatch_bf16_combine_sm_free"]["overlap_compute_coverage_pct"],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
