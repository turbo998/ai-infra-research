import unittest
from experiments.llm_optimizer_param_group_weight_decay_rmsnorm_muon_1d_2d_routing_selection.run import evaluate_optimizer_param_group_routing


class TestOptimizerParamGroupRouting(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_optimizer_param_group_routing()
        self.assertEqual(res["round"], 323)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_three_way_hybrid_muon_2d_hidden_plus_adamw_embed_plus_zero_wd_1d"]["validation_perplexity_ppl"],
            6.8,
        )
        self.assertGreater(
            tiers["tier4_three_way_hybrid_muon_2d_hidden_plus_adamw_embed_plus_zero_wd_1d"]["steps_to_target_loss_speedup_x"],
            1.35,
        )


if __name__ == "__main__":
    unittest.main()
