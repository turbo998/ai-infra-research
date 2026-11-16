import unittest
from experiments.llm_rlvr_code_math_verifier_sandbox_async_reward_pipeline_selection.run import evaluate_rlvr_verifier_sandbox_pipeline


class TestRlvrVerifierSandboxPipeline(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_rlvr_verifier_sandbox_pipeline()
        self.assertEqual(res["round"], 300)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_streaming_per_sample_nsjail_sandbox_plus_group_ready_grpo_trigger"]["gpu_verifier_wait_stall_sec"],
            2.0,
        )
        self.assertGreater(
            tiers["tier4_streaming_per_sample_nsjail_sandbox_plus_group_ready_grpo_trigger"]["effective_gpu_utilization_pct"],
            85.0,
        )


if __name__ == "__main__":
    unittest.main()
