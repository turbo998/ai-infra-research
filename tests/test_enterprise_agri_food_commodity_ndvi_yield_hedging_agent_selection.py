from __future__ import annotations

import unittest
from experiments.enterprise_agri_food_commodity_ndvi_yield_hedging_agent_selection.run import build_payload


class TestEnterpriseAgriFoodCommodityNdviYieldHedgingAgentSelection(unittest.TestCase):
    def test_tier4_yield_mape_sar_cloud_penetration_and_cvar(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["regional_yield_forecast_mape_pct"], 3.0)
        self.assertEqual(tier4["monsoon_cloud_blindness_gap_days"], 0.0)
        self.assertLessEqual(tier4["unhedged_tail_price_spike_loss_pct"], 2.0)
        self.assertEqual(tier4["silo_spoilage_mycotoxin_escape_pct"], 0.0)
        self.assertLess(tier4["monthly_procurement_variance_and_spoilage_usd"], tier1["monthly_procurement_variance_and_spoilage_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
