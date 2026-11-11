"""Unit tests for Round 53 Long-Context KV Cache Quantization & Dynamic Sparse Attention Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.long_context_kv_cache_compression_quantization_selection.run import (
    evaluate_long_context_kv_cache_compression_matrix,
)


class TestLongContextKvCacheCompressionQuantizationSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_long_context_kv_cache_compression_matrix()
        self.assertEqual(data["experiment_id"], "long_context_kv_cache_compression_quantization_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_fp8_kv_plus_quest_dynamic_page_sparse_attention"]["decode_speed_tok_s_per_req"],
            tiers["tier1_uncompressed_bf16_full_kv_cache"]["decode_speed_tok_s_per_req"] * 5.5,
        )
        self.assertGreater(
            tiers["tier4_fp8_kv_plus_quest_dynamic_page_sparse_attention"]["multi_turn_ruler_accuracy_pct"],
            tiers["tier2_permanent_token_eviction_h2o_snapkv"]["multi_turn_ruler_accuracy_pct"] + 35.0,
        )
        self.assertGreater(
            data["summary"]["tier4_cost_reduction_vs_tier1_pct"],
            80.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "long_context_kv_cache_compression_quantization_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_long_context_kv_cache_compression_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
