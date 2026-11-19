from __future__ import annotations

import unittest
from experiments.enterprise_mining_mineral_processing_haulage_mill_mpc_tailings_agent_selection.run import build_payload


class TestEnterpriseMiningMineralProcessingHaulageMillMpcTailingsAgentSelection(unittest.TestCase):
    def test_tier4_mill_stalls_haulage_waste_and_zero_tailings_escape(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["sag_mill_overload_and_haulage_stall_events_per_month"], 0.5)
        self.assertLessEqual(tier4["haul_truck_queue_and_shovel_starvation_waste_pct"], 2.0)
        self.assertLessEqual(tier4["grinding_specific_energy_and_recovery_loss_pct"], 1.5)
        self.assertEqual(tier4["tailings_dam_seepage_and_slope_alarm_escape_pct"], 0.0)
        self.assertLess(
            tier4["monthly_haulage_mill_and_recovery_loss_usd"],
            tier1["monthly_haulage_mill_and_recovery_loss_usd"] * 0.12,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
