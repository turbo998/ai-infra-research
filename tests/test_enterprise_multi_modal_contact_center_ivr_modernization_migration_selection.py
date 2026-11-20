from __future__ import annotations

import unittest
from experiments.enterprise_multi_modal_contact_center_ivr_modernization_migration_selection.run import build_payload


class TestEnterpriseMultiModalContactCenterIvrModernizationMigrationSelection(unittest.TestCase):
    def test_tier4_maximizes_containment_preserves_pci_and_minimizes_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreater(tier4["self_service_containment_rate_pct"], 84.0)
        self.assertGreater(
            tier4["self_service_containment_rate_pct"],
            tier1["self_service_containment_rate_pct"] + 50.0,
        )
        self.assertEqual(tier4["pci_dss_card_capture_compliance_pct"], 100.0)
        self.assertGreater(
            tier4["pci_dss_card_capture_compliance_pct"],
            tier2["pci_dss_card_capture_compliance_pct"] + 90.0,
        )
        self.assertLess(
            tier4["voice_turn_around_p95_latency_ms"],
            tier3["voice_turn_around_p95_latency_ms"] * 0.25,
        )
        self.assertLess(
            tier4["monthly_5m_call_contact_center_tco_usd"],
            tier1["monthly_5m_call_contact_center_tco_usd"] * 0.30,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
