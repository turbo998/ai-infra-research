"""Unit tests for Round 71 Enterprise LLM Observability, OTel GenAI & Tail-Based Drift Sampling Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.llm_observability_otel_genai_tail_sampling_selection.run import (
    evaluate_llm_observability_otel_genai_tail_sampling_matrix,
)


class TestLlmObservabilityOtelGenaiTailSamplingSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_llm_observability_otel_genai_tail_sampling_matrix()
        self.assertEqual(
            data["experiment_id"],
            "llm_observability_otel_genai_tail_sampling_selection",
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_otel_genai_tail_based_anomaly_sampling_plus_drift_pii"]["silent_failure_capture_recall_pct"],
            90.0,
        )
        self.assertLess(
            tiers["tier4_otel_genai_tail_based_anomaly_sampling_plus_drift_pii"]["judge_evaluated_traffic_pct"],
            10.0,
        )
        self.assertGreater(
            data["summary"]["tier4_vs_tier3_failure_recall_gain_pts"],
            70.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "llm_observability_otel_genai_tail_sampling_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_llm_observability_otel_genai_tail_sampling_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
