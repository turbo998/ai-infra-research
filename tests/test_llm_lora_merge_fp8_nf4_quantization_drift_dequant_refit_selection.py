import unittest
from experiments.llm_lora_merge_fp8_nf4_quantization_drift_dequant_refit_selection.run import evaluate_lora_quant_merge


class TestLoraQuantMerge(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_lora_quant_merge()
        self.assertEqual(res["round"], 283)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_loftq_or_bf16_master_merge_plus_sft_calibrated_awq_fp8"]["domain_accuracy_pct"]
            - tiers["tier1_naive_peft_merge_into_nf4_requant_int4"]["domain_accuracy_pct"],
            8.0,
        )
        self.assertGreater(
            tiers["tier4_loftq_or_bf16_master_merge_plus_sft_calibrated_awq_fp8"]["decode_throughput_tok_s"],
            tiers["tier2_merge_into_original_bf16_master_no_quant"]["decode_throughput_tok_s"] * 2.5,
        )


if __name__ == "__main__":
    unittest.main()
