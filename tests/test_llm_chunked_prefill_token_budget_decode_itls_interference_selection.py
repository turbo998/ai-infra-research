import unittest
from experiments.llm_chunked_prefill_token_budget_decode_itls_interference_selection.run import (
    evaluate_chunked_prefill_token_budget_tiers,
)


class TestChunkedPrefillTokenBudgetSelection(unittest.TestCase):
    def test_chunked_prefill_tiers(self) -> None:
        data = evaluate_chunked_prefill_token_budget_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_oversized_chunk_16384_decode_itl_freeze"]
        t2 = tiers["tier2_undersized_chunk_256_quadratic_kv_reread_tax"]
        t4 = tiers["tier4_slo_aware_dynamic_chunk_budget_plus_long_prompt_isolation"]
        self.assertGreater(t1["p99_decode_itl_jitter_ms"], 600.0)
        self.assertGreater(t2["kv_cache_hbm_reread_multiplier_x"], 60.0)
        self.assertLess(t4["p99_decode_itl_jitter_ms"], 30.0)
        self.assertLess(t4["prompt_32k_ttft_ms"], 1500.0)


if __name__ == "__main__":
    unittest.main()
