"""Unit tests for Round 178 LLM Serving Engine CPU Scheduling Overhead & Zero-Overhead Async Multiprocess Architecture Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.llm_serving_cpu_overhead_zeromq_async_scheduler_selection.run import (
    evaluate_llm_serving_cpu_overhead_zeromq_async_scheduler_matrix,
)


class TestLlmServingCpuOverheadZeromqAsyncSchedulerSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_llm_serving_cpu_overhead_zeromq_async_scheduler_matrix()
        self.assertEqual(
            data["experiment_id"],
            "llm_serving_cpu_overhead_zeromq_async_scheduler_selection",
        )
        self.assertEqual(data["round"], 178)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLess(
            tiers["tier4_zero_overhead_overlap_scheduler_future_token"]["exposed_cpu_overhead_ms"],
            0.15,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_cpu_overhead_reduction_ratio"],
            70.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier1_throughput_speedup_ratio"],
            2.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "llm_serving_cpu_overhead_zeromq_async_scheduler_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_llm_serving_cpu_overhead_zeromq_async_scheduler_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
