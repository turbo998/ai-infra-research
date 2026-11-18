from __future__ import annotations

import unittest
from experiments.enterprise_railway_rolling_stock_catenary_track_maintenance_agent_selection.run import build_payload


class TestEnterpriseRailwayRollingStockCatenaryTrackMaintenanceAgentSelection(unittest.TestCase):
    def test_tier4_disruption_reduction_false_stops_and_zero_defect_escape(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["mainline_service_disruption_events_per_month"], 0.5)
        self.assertLessEqual(tier4["false_positive_train_stop_rate_pct"], 0.5)
        self.assertLessEqual(tier4["wheelset_shelling_and_lathe_waste_pct"], 2.0)
        self.assertEqual(tier4["catenary_dewire_and_rail_defect_escape_pct"], 0.0)
        self.assertLess(tier4["monthly_delay_wheel_and_possession_cost_usd"], tier1["monthly_delay_wheel_and_possession_cost_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
