from __future__ import annotations

import unittest
from experiments.enterprise_datacenter_dcim_stranded_power_srlg_fiber_capacity_milp_selection.run import build_payload


class TestEnterpriseDatacenterDcimStrandedPowerSrlgFiberCapacityMilpSelection(unittest.TestCase):
    def test_tier4_stranded_power_zero_srlg_and_zero_pdu_trip(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["stranded_unusable_datacenter_power_mw_pct"], 2.5)
        self.assertEqual(tier4["srlg_shared_conduit_dual_path_violation_pct"], 0.0)
        self.assertEqual(tier4["three_phase_pdu_current_imbalance_trip_risk_pct"], 0.0)
        self.assertEqual(tier4["dac_aoc_cable_reach_violation_rework_pct"], 0.0)
        self.assertLess(
            tier4["monthly_stranded_capex_and_provisioning_loss_usd"],
            tier1["monthly_stranded_capex_and_provisioning_loss_usd"] * 0.10,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
