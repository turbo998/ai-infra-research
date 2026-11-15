import unittest
from experiments.llm_flashattention_sinks_fp8_online_softmax_quantization_drift_selection.run import evaluate_fa3_fp8_incoherent_block_quant


class TestFa3Fp8IncoherentBlockQuant(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_fa3_fp8_incoherent_block_quant()
        self.assertEqual(res["round"], 295)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_naive_per_tensor_fp8_fa3_sink_outlier_crush"]["output_rmse_vs_fp32"],
            tiers["tier4_fa3_per_block_fp8_plus_incoherent_hadamard_qh_kh_rotation"]["output_rmse_vs_fp32"] * 15.0,
        )
        self.assertGreater(
            tiers["tier4_fa3_per_block_fp8_plus_incoherent_hadamard_qh_kh_rotation"]["long_context_acc_pct"],
            84.0,
        )


if __name__ == "__main__":
    unittest.main()
