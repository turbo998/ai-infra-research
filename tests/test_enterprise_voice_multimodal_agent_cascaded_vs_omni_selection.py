"""Unit tests for Round 376 Enterprise Voice AI Agent & Multimodal Real-Time Architecture Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.enterprise_voice_multimodal_agent_cascaded_vs_omni_selection.run import (
    evaluate_enterprise_voice_multimodal_agent_cascaded_vs_omni,
)


class TestEnterpriseVoiceMultimodalAgentCascadedVsOmniSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_enterprise_voice_multimodal_agent_cascaded_vs_omni()
        self.assertEqual(
            data["experiment_id"],
            "enterprise_voice_multimodal_agent_cascaded_vs_omni_selection",
        )
        self.assertEqual(data["round"], 376)
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertLessEqual(
            tiers["tier4_hybrid_duplex_speculative_rag_emotion_tags_selective_omni"]["p99_ttfa_ms"],
            400.0,
        )
        self.assertGreater(
            data["summary"]["tier4_cost_savings_vs_tier2_omni_pct"],
            80.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "enterprise_voice_multimodal_agent_cascaded_vs_omni_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_enterprise_voice_multimodal_agent_cascaded_vs_omni()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
