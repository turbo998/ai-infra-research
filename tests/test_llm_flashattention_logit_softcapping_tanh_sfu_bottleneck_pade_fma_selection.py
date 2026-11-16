import unittest
from experiments.llm_flashattention_logit_softcapping_tanh_sfu_bottleneck_pade_fma_selection.run import evaluate_flashattention_softcapping_tanh


class TestFlashAttentionSoftcappingTanh(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_flashattention_softcapping_tanh()
        self.assertEqual(res["round"], 314)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_rational_pade_fma_tanh_emulation_plus_folded_cap_log2e_scale"]["effective_attention_tflops"],
            550.0,
        )
        self.assertLess(
            tiers["tier4_rational_pade_fma_tanh_emulation_plus_folded_cap_log2e_scale"]["max_attention_output_abs_err"],
            1e-4,
        )


if __name__ == "__main__":
    unittest.main()
