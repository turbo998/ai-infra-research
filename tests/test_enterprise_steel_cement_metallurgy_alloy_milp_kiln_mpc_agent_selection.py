from __future__ import annotations

import unittest
from experiments.enterprise_steel_cement_metallurgy_alloy_milp_kiln_mpc_agent_selection.run import build_payload


class TestEnterpriseSteelCementMetallurgyAlloyMilpKilnMpcAgentSelection(unittest.TestCase):
    def test_tier4_alloy_savings_zero_downgrade_and_kiln_mpc(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["ferroalloy_overconsumption_waste_pct"], 1.5)
        self.assertEqual(tier4["off_spec_heat_downgrade_rate_pct"], 0.0)
        self.assertLessEqual(tier4["kiln_specific_fuel_overburn_pct"], 1.5)
        self.assertLessEqual(tier4["ladle_refractory_breakout_or_early_waste_pct"], 2.0)
        self.assertLess(tier4["monthly_alloy_fuel_and_downgrade_loss_usd"], tier1["monthly_alloy_fuel_and_downgrade_loss_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
