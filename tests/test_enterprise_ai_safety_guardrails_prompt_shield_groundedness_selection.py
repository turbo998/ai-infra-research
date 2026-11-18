"""Unit tests for Enterprise AI Safety Guardrails, Prompt Shield & Streaming Groundedness Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_ai_safety_guardrails_prompt_shield_groundedness_selection.run import (
    evaluate_enterprise_ai_safety_guardrails_prompt_shield_groundedness,
)


class TestEnterpriseAiSafetyGuardrailsPromptShieldGroundednessSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_ai_safety_guardrails_prompt_shield_groundedness()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_ai_safety_guardrails_prompt_shield_groundedness_selection",
        )
        self.assertEqual(data["round"], 377)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_defense_in_depth_spotlighting_parallel_shield_nli_opa"]["attack_interception_rate_pct"],
            99.0,
        )
        self.assertLessEqual(
            tiers["tier4_defense_in_depth_spotlighting_parallel_shield_nli_opa"]["added_p99_ttft_penalty_ms"],
            20.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_ai_safety_guardrails_prompt_shield_groundedness_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_ai_safety_guardrails_prompt_shield_groundedness()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
