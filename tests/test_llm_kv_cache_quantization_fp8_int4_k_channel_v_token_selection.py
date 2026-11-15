import unittest
from experiments.llm_kv_cache_quantization_fp8_int4_k_channel_v_token_selection.run import (
    evaluate_kv_cache_quantization_asymmetry_tiers,
)


class TestKvCacheQuantizationAsymmetrySelection(unittest.TestCase):
    def test_kv_quant_tiers(self) -> None:
        data = evaluate_kv_cache_quantization_asymmetry_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_per_token_int4_on_both_key_and_value"]
        t4 = tiers["tier4_hadamard_rotated_int4_kv_plus_fp16_sink_and_recent_window"]
        self.assertLess(t1["long_context_ruler_accuracy_pct"], 65.0)
        self.assertGreater(t4["long_context_ruler_accuracy_pct"], 91.0)
        self.assertGreater(t4["kv_cache_memory_compression_x"], 3.8)


if __name__ == "__main__":
    unittest.main()
