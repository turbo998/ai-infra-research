from __future__ import annotations

import unittest
from experiments.enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection.run import build_payload


class TestEnterpriseMaritimeShippingWeatherRoutingPortDemurrageAgentSelection(unittest.TestCase):
    def test_tier4_bunker_savings_jit_eta_and_laytime_fsm(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["bunker_fuel_and_eu_ets_overburn_pct"], 1.5)
        self.assertLessEqual(tier4["port_berth_eta_mae_hours"], 2.0)
        self.assertEqual(tier4["demurrage_laytime_overpayment_rate_pct"], 0.0)
        self.assertEqual(tier4["imo_cii_rating_downgrade_risk_pct"], 0.0)
        self.assertLess(tier4["monthly_bunker_and_demurrage_waste_usd"], tier1["monthly_bunker_and_demurrage_waste_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
