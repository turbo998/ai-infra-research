from __future__ import annotations

import unittest
from experiments.enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection.run import build_payload


class TestEnterpriseAutomotiveAftermarketPartsCrostonCollisionVlmBayMilpSelection(unittest.TestCase):
    def test_tier4_vor_stockout_obsolescence_and_zero_wrong_vin_fitment(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["vor_vehicle_off_road_parts_stockout_pct"], 0.8)
        self.assertLessEqual(tier4["obsolete_dead_stock_inventory_writeoff_pct"], 2.0)
        self.assertLessEqual(tier4["collision_supplement_rework_and_part_return_pct"], 2.0)
        self.assertEqual(tier4["wrong_vin_part_fitment_escape_pct"], 0.0)
        self.assertLess(
            tier4["monthly_parts_stockout_obsolescence_and_bay_loss_usd"],
            tier1["monthly_parts_stockout_obsolescence_and_bay_loss_usd"] * 0.12,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
