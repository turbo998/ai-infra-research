from __future__ import annotations

import unittest
from experiments.enterprise_long_context_window_vs_rag_memory_tiering_selection.run import build_payload


class TestEnterpriseLongContextWindowVsRagMemoryTieringSelection(unittest.TestCase):
    def test_tier4_highest_multi_needle_and_turn50_memory_at_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertGreater(tier4["multi_needle_reasoning_accuracy_pct"], 97.0)
        self.assertGreater(
            tier4["multi_needle_reasoning_accuracy_pct"],
            tier1["multi_needle_reasoning_accuracy_pct"] + 20.0,
        )
        self.assertGreater(tier4["long_session_turn_50_memory_recall_pct"], 98.0)
        self.assertGreater(
            tier4["long_session_turn_50_memory_recall_pct"],
            tier2["long_session_turn_50_memory_recall_pct"] + 65.0,
        )
        self.assertLess(tier4["p95_ttft_latency_ms"], tier1["p95_ttft_latency_ms"] * 0.10)
        self.assertLess(
            tier4["monthly_llm_context_and_memory_tco_usd"],
            tier1["monthly_llm_context_and_memory_tco_usd"] * 0.10,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
