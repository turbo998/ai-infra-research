import unittest
from experiments.llm_async_rl_rollout_trainer_disaggregation_ray_staleness_selection.run import evaluate_async_rl_disaggregation


class TestAsyncRlDisaggregation(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_async_rl_disaggregation()
        self.assertEqual(res["round"], 282)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_sync_colocated_ray_object_store_weight_sync"]["weight_sync_latency_s"],
            tiers["tier4_bounded_async_tau1_partial_rollout_nccl_rdma_dual_clip"]["weight_sync_latency_s"] * 20.0,
        )
        self.assertGreater(
            tiers["tier4_bounded_async_tau1_partial_rollout_nccl_rdma_dual_clip"]["steps_per_hour"],
            tiers["tier1_sync_colocated_ray_object_store_weight_sync"]["steps_per_hour"] * 2.5,
        )


if __name__ == "__main__":
    unittest.main()
