from __future__ import annotations

import unittest
from experiments.enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection.run import build_payload


class TestEnterpriseAgenticRagVsStandardRagRouterQueryDecompositionSelection(unittest.TestCase):
    def test_tier4_achieves_highest_accuracy_with_sub1200ms_latency_and_low_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreater(tier4["multi_hop_complex_query_accuracy_pct"], 94.0)
        self.assertGreater(
            tier4["multi_hop_complex_query_accuracy_pct"],
            tier1["multi_hop_complex_query_accuracy_pct"] + 45.0,
        )
        self.assertLess(
            tier4["blended_p95_response_latency_ms"],
            tier2["blended_p95_response_latency_ms"] * 0.15,
        )
        self.assertLess(
            tier4["monthly_30m_query_platform_tco_usd"],
            tier2["monthly_30m_query_platform_tco_usd"] * 0.25,
        )
        self.assertGreater(
            tier4["blended_answer_accuracy_pct"],
            tier3["blended_answer_accuracy_pct"] + 8.0,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
