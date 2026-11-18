from __future__ import annotations

import unittest
from experiments.enterprise_water_utility_leak_detection_wastewater_aeration_mpc_agent_selection.run import build_payload


class TestEnterpriseWaterUtilityLeakDetectionWastewaterAerationMpcAgentSelection(unittest.TestCase):
    def test_tier4_nrw_reduction_aeration_savings_and_zero_permit_breach(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["non_revenue_water_loss_pct"], 8.0)
        self.assertLessEqual(tier4["aeration_blower_and_chemical_overuse_pct"], 2.5)
        self.assertEqual(tier4["effluent_ammonia_phosphorus_permit_breach_pct"], 0.0)
        self.assertLessEqual(tier4["pipe_burst_localization_radius_meters"], 40.0)
        self.assertLess(tier4["monthly_energy_chemical_and_water_loss_usd"], tier1["monthly_energy_chemical_and_water_loss_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
