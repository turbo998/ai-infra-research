from __future__ import annotations

import unittest
from experiments.enterprise_media_gaming_liveops_localization_compliance_agent_selection.run import build_payload


class TestEnterpriseMediaGamingLiveopsLocalizationComplianceAgentSelection(unittest.TestCase):
    def test_tier4_simship_isochrony_and_compliance_gate(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["lore_terminology_violation_rate_pct"], 0.0)
        self.assertLessEqual(tier4["dubbing_isochrony_desync_rate_pct"], 0.5)
        self.assertEqual(tier4["regional_age_rating_violation_escape_pct"], 0.0)
        self.assertLessEqual(tier4["global_patch_launch_lag_hours"], 3.0)
        self.assertLess(tier4["monthly_localization_and_promo_waste_usd"], tier1["monthly_localization_and_promo_waste_usd"] * 0.15)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
