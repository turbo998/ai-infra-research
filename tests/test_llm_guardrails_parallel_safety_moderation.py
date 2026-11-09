from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.llm_guardrails_parallel_safety_moderation.run import (
    evaluate_llm_guardrail_tiers,
)


class TestLlmGuardrailsParallelSafetyModeration(unittest.TestCase):
    def test_provenance_and_guardrail_invariants(self) -> None:
        data = evaluate_llm_guardrail_tiers()
        self.assertEqual(
            data["experiment_id"], "llm_guardrails_parallel_safety_moderation"
        )
        self.assertFalse(data["provenance"]["independently_reproduced"])
        self.assertEqual(
            data["provenance"]["physical_gpu_cluster_measurement"], "not_measured"
        )

        summary = data["summary"]
        self.assertEqual(summary["tier2_ttft_overhead_ms"], 580.0)
        self.assertEqual(summary["tier3_ttft_overhead_ms"], 8.0)
        self.assertEqual(summary["tier4_ttft_overhead_ms"], 1.5)
        self.assertEqual(summary["tier4_jailbreak_recall_pct"], 98.4)

    def test_results_json_matches_deterministic_model(self) -> None:
        results_path = (
            Path(__file__).resolve().parent.parent
            / "experiments"
            / "llm_guardrails_parallel_safety_moderation"
            / "results.json"
        )
        on_disk = json.loads(results_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, evaluate_llm_guardrail_tiers())


if __name__ == "__main__":
    unittest.main()
