"""Unit tests for Round 184 Long-Context KV Cache Budget Control, Attention Sinks & DuoAttention Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.attention_sink_snapkv_duoattention_kv_eviction_selection.run import (
    evaluate_attention_sink_snapkv_duoattention_kv_eviction_matrix,
)


class TestAttentionSinkSnapkvDuoattentionKvEvictionSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_attention_sink_snapkv_duoattention_kv_eviction_matrix()
        self.assertEqual(
            data["experiment_id"],
            "attention_sink_snapkv_duoattention_kv_eviction_selection",
        )
        self.assertEqual(data["round"], 184)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier1_naive_sliding_window_no_sinks"]["streaming_perplexity"],
            1000.0,
        )
        self.assertGreater(
            tiers["tier4_duoattention_snapkv_retrieval_head_split"]["niah_128k_recall_pct"],
            98.0,
        )
        self.assertGreater(
            data["summary"]["tier4_kv_memory_reduction_vs_full_40gb_pct"],
            70.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "attention_sink_snapkv_duoattention_kv_eviction_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_attention_sink_snapkv_duoattention_kv_eviction_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
