import unittest
from experiments.kv_cache_quantization_kivi_kvquant_prerope_selection.run import (
    evaluate_kv_quantization_tiers,
)


class TestKvQuantizationKiviKvquantSelection(unittest.TestCase):
    def test_kv_quant_tiers(self) -> None:
        data = evaluate_kv_quantization_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t2 = tiers["tier2_naive_per_token_symmetric_int4_kv"]
        t3 = tiers["tier3_kivi_asymmetric_per_channel_k_per_token_v"]
        t4 = tiers["tier4_kvquant_prerope_key_plus_1pct_outlier_nuq"]
        self.assertGreater(t2["wikitext2_perplexity"], 20.0)
        self.assertLess(t3["wikitext2_perplexity"], 5.3)
        self.assertGreaterEqual(t4["needle_in_haystack_128k_acc_pct"], 99.0)


if __name__ == "__main__":
    unittest.main()
