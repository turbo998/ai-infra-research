"""Unit tests for Round 368 Real-Time Customer Service & Enterprise Copilot Sub-400ms Low-Latency Architecture Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.realtime_customer_service_copilot_low_latency_selection.run import (
    evaluate_realtime_customer_service_copilot_low_latency,
)


class TestRealtimeCustomerServiceCopilotLowLatencySelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_realtime_customer_service_copilot_low_latency()
        self.assertEqual(
            data["experiment_id"],
            "realtime_customer_service_copilot_low_latency_selection",
        )
        self.assertEqual(data["round"], 368)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLessEqual(
            tiers["tier4_speculative_rag_prefetch_optimistic_prefill_eagle3_duplex"]["p99_e2e_ttft_ms"],
            400.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_p99_ttft_speedup_x"],
            10.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "realtime_customer_service_copilot_low_latency_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_realtime_customer_service_copilot_low_latency()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
