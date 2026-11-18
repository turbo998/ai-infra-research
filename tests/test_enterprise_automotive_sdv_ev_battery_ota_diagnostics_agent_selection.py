from __future__ import annotations

import unittest
from experiments.enterprise_automotive_sdv_ev_battery_ota_diagnostics_agent_selection.run import build_payload


class TestEnterpriseAutomotiveSdvEvBatteryOtaDiagnosticsAgentSelection(unittest.TestCase):
    def test_tier4_early_warning_ntf_reduction_and_ota_gate(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["thermal_runaway_early_warning_days"], 20.0)
        self.assertLessEqual(tier4["warranty_no_trouble_found_ntf_rate_pct"], 3.0)
        self.assertEqual(tier4["bad_ota_fleet_brick_exposure_pct"], 0.0)
        self.assertLessEqual(tier4["cellular_telemetry_gb_per_vehicle_month"], 1.0)
        self.assertLess(tier4["monthly_cellular_and_warranty_recall_cost_usd"], tier1["monthly_cellular_and_warranty_recall_cost_usd"] * 0.15)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
