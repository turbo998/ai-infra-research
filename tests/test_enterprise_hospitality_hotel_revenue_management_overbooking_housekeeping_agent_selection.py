from __future__ import annotations

import unittest
from experiments.enterprise_hospitality_hotel_revenue_management_overbooking_housekeeping_agent_selection.run import build_payload


class TestEnterpriseHospitalityHotelRevenueManagementOverbookingHousekeepingAgentSelection(unittest.TestCase):
    def test_tier4_revpar_uplift_walk_risk_and_housekeeping_milp(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["revpar_uplift_vs_baseline_pct"], 14.0)
        self.assertLessEqual(tier4["guest_walk_or_empty_spoilage_rate_pct"], 0.5)
        self.assertLessEqual(tier4["early_checkin_wait_breach_pct"], 1.0)
        self.assertLess(tier4["monthly_spoilage_walk_and_overtime_cost_usd"], tier1["monthly_spoilage_walk_and_overtime_cost_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
