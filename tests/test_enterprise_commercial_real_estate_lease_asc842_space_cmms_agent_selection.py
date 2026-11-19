from __future__ import annotations

import unittest
from experiments.enterprise_commercial_real_estate_lease_asc842_space_cmms_agent_selection.run import build_payload


class TestEnterpriseCommercialRealEstateLeaseAsc842SpaceCmmsAgentSelection(unittest.TestCase):
    def test_tier4_cam_recovery_zero_asc842_error_and_zero_warranty_leakage(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["cam_overcharge_and_missed_break_option_loss_pct"], 0.2)
        self.assertEqual(tier4["asc842_ifrs16_rou_liability_schedule_error_pct"], 0.0)
        self.assertLessEqual(tier4["unutilized_ghost_floor_space_and_hvac_waste_pct"], 5.0)
        self.assertEqual(tier4["warranty_covered_cmms_vendor_double_billing_pct"], 0.0)
        self.assertLess(
            tier4["monthly_cre_lease_cam_and_facility_waste_usd"],
            tier1["monthly_cre_lease_cam_and_facility_waste_usd"] * 0.12,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
