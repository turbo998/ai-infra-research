from __future__ import annotations

import unittest
from experiments.enterprise_hybrid_cloud_private_ai_landing_zone_selection.run import build_payload


class TestEnterpriseHybridCloudPrivateAiLandingZoneSelection(unittest.TestCase):
    def test_tier4_full_sovereignty_high_capability_low_latency_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertEqual(tier4["l4_restricted_data_sovereignty_compliance_pct"], 100.0)
        self.assertGreater(tier4["frontier_model_capability_coverage_pct"], 99.0)
        self.assertGreater(
            tier4["frontier_model_capability_coverage_pct"],
            tier2["frontier_model_capability_coverage_pct"] + 30.0,
        )
        self.assertLess(
            tier4["hybrid_cross_premises_rag_p95_latency_ms"],
            tier3["hybrid_cross_premises_rag_p95_latency_ms"] * 0.20,
        )
        self.assertLess(
            tier4["monthly_amortized_hybrid_infra_tco_usd"],
            tier2["monthly_amortized_hybrid_infra_tco_usd"] * 0.30,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
