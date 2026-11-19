from __future__ import annotations

import unittest
from experiments.enterprise_llm_guardrails_prompt_shield_content_safety_latency_selection.run import build_payload


class TestEnterpriseLlmGuardrailsPromptShieldContentSafetyLatencySelection(unittest.TestCase):
    def test_tier4_highest_safety_and_sub_20ms_ttft_overhead(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertGreater(tier4["jailbreak_and_indirect_injection_block_rate_pct"], 99.0)
        self.assertGreater(tier4["ungrounded_rag_hallucination_catch_rate_pct"], 97.0)
        self.assertLessEqual(tier4["added_p99_ttft_guardrail_overhead_ms"], 20.0)
        self.assertLess(
            tier4["added_p99_ttft_guardrail_overhead_ms"],
            tier2["added_p99_ttft_guardrail_overhead_ms"] * 0.01,
        )
        self.assertLess(tier4["false_positive_benign_refusal_rate_pct"], tier1["false_positive_benign_refusal_rate_pct"])

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
