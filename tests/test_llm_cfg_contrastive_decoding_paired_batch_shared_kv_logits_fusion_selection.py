import unittest
from experiments.llm_cfg_contrastive_decoding_paired_batch_shared_kv_logits_fusion_selection.run import evaluate_cfg_contrastive_decoding_serving


class TestCfgContrastiveDecodingServing(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_cfg_contrastive_decoding_serving()
        self.assertEqual(res["round"], 312)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_atomic_paired_sequence_group_shared_prefix_plus_fused_triton_cfg_sampler"]["decode_step_tpot_ms"],
            22.5,
        )
        self.assertGreater(
            tiers["tier4_atomic_paired_sequence_group_shared_prefix_plus_fused_triton_cfg_sampler"]["serving_throughput_tok_per_sec"],
            2800.0,
        )


if __name__ == "__main__":
    unittest.main()
