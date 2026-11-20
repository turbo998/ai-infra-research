from __future__ import annotations

import unittest
from experiments.enterprise_multimodal_customer_service_copilot_architecture_sizing_selection.run import build_payload


class TestEnterpriseMultimodalCustomerServiceCopilotArchitectureSizingSelection(unittest.TestCase):
    def test_tier4_achieves_sub400ms_ttft_highest_fcr_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["end_to_end_ttft_p99_latency_ms"], 400.0)
        self.assertLess(
            tier4["end_to_end_ttft_p99_latency_ms"],
            tier1["end_to_end_ttft_p99_latency_ms"] * 0.15,
        )
        self.assertGreater(tier4["first_contact_resolution_fcr_pct"], 93.0)
        self.assertGreater(
            tier4["first_contact_resolution_fcr_pct"],
            tier2["first_contact_resolution_fcr_pct"] + 25.0,
        )
        self.assertLess(
            tier4["monthly_60m_turn_platform_tco_usd"],
            tier3["monthly_60m_turn_platform_tco_usd"] * 0.35,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
