"""Unit tests for Round 54 LLM Request Scheduling, Chunked Prefill & SLO Preemption Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.llm_request_scheduling_chunked_prefill_slo_selection.run import (
    evaluate_llm_request_scheduling_chunked_prefill_matrix,
)


class TestLlmRequestSchedulingChunkedPrefillSloSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_llm_request_scheduling_chunked_prefill_matrix()
        self.assertEqual(data["experiment_id"], "llm_request_scheduling_chunked_prefill_slo_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier2_prefill_prioritized_continuous_recompute_preemption"]["p99_tpot_ms"],
            tiers["tier3_sarathi_chunked_prefill_piggyback_dram_swap"]["p99_tpot_ms"] * 40.0,
        )
        self.assertLess(
            tiers["tier4_vllm_v1_zero_overhead_async_scheduler_slo_watermark"]["p99_tpot_ms"],
            30.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier2_goodput_speedup_ratio"],
            3.5,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "llm_request_scheduling_chunked_prefill_slo_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_llm_request_scheduling_chunked_prefill_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
