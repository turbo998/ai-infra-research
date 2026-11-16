import unittest
from experiments.llm_large_vocab_fused_linear_cross_entropy_liger_cce_memory_selection.run import evaluate_large_vocab_fused_linear_ce


class TestLargeVocabFusedLinearCe(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_large_vocab_fused_linear_ce()
        self.assertEqual(res["round"], 301)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_cut_cross_entropy_triton_indexed_dot_plus_chunked_fused_bwd"]["logits_ce_peak_memory_gb"],
            0.5,
        )
        self.assertGreater(
            tiers["tier4_cut_cross_entropy_triton_indexed_dot_plus_chunked_fused_bwd"]["training_tokens_per_sec"],
            30000.0,
        )


if __name__ == "__main__":
    unittest.main()
