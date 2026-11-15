import unittest
from experiments.llm_cross_node_speculative_decoding_eagle3_draft_placement_selection.run import evaluate_eagle3_draft_placement


class TestEagle3DraftPlacement(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_eagle3_draft_placement()
        self.assertEqual(res["round"], 285)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_standalone_7b_draft_sharded_tp8_280_allreduces"]["draft_5step_latency_ms"],
            tiers["tier4_eagle3_multilayer_fusion_pruned_32k_vocab_joint_cudagraph"]["draft_5step_latency_ms"] * 9.0,
        )
        self.assertLess(
            tiers["tier4_eagle3_multilayer_fusion_pruned_32k_vocab_joint_cudagraph"]["effective_tpot_ms"],
            3.5,
        )


if __name__ == "__main__":
    unittest.main()
