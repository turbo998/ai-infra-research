from __future__ import annotations

import unittest
from experiments.enterprise_rag_hallucination_detection_citation_grounding_verification_selection.run import build_payload


class TestEnterpriseRagHallucinationDetectionCitationGroundingVerificationSelection(unittest.TestCase):
    def test_tier4_minimizes_hallucinations_and_citation_errors_with_sub30ms_overhead(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["ungrounded_hallucination_rate_pct"], 0.2)
        self.assertLessEqual(tier4["fake_or_mismatched_citation_rate_pct"], 0.1)
        self.assertLess(
            tier4["verification_latency_p95_overhead_ms"],
            tier2["verification_latency_p95_overhead_ms"] * 0.02,
        )
        self.assertGreater(
            tier4["numeric_and_entity_exact_match_pct"],
            tier3["numeric_and_entity_exact_match_pct"] + 15.0,
        )
        self.assertLess(
            tier4["monthly_25m_query_verification_and_risk_tco_usd"],
            tier2["monthly_25m_query_verification_and_risk_tco_usd"] * 0.12,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
