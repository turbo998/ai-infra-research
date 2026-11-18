from __future__ import annotations

import unittest
from experiments.enterprise_warehouse_robotics_amr_mapf_cubing_waveless_agent_selection.run import build_payload


class TestEnterpriseWarehouseRoboticsAmrMapfCubingWavelessAgentSelection(unittest.TestCase):
    def test_tier4_zero_deadlock_waveless_uph_and_3d_cubing(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertEqual(tier4["amr_aisle_deadlock_and_congestion_pct"], 0.0)
        self.assertLessEqual(tier4["pick_station_starvation_idle_pct"], 1.5)
        self.assertLessEqual(tier4["carton_void_air_shipping_waste_pct"], 7.0)
        self.assertGreaterEqual(tier4["units_picked_per_station_hour_uph"], 480)
        self.assertLess(tier4["monthly_fulfillment_and_dim_shipping_loss_usd"], tier1["monthly_fulfillment_and_dim_shipping_loss_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
