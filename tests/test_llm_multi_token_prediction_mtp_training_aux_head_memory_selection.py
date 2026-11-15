import unittest
from experiments.llm_multi_token_prediction_mtp_training_aux_head_memory_selection.run import evaluate_mtp_training_and_serving


class TestMtpTrainingAndServing(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_mtp_training_and_serving()
        self.assertEqual(res["round"], 280)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_naive_parallel_independent_mtp_heads_full_logits"]["peak_logits_hbm_gb"],
            tiers["tier4_causal_mtp_shared_head_plus_fused_linear_ce_plus_tree_verify"]["peak_logits_hbm_gb"] * 100.0,
        )
        self.assertGreater(
            tiers["tier4_causal_mtp_shared_head_plus_fused_linear_ce_plus_tree_verify"]["decode_speedup_x"],
            2.4,
        )


if __name__ == "__main__":
    unittest.main()
