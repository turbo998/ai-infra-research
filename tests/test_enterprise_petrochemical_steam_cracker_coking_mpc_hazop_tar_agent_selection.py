from __future__ import annotations

import unittest
from experiments.enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection.run import build_payload


class TestEnterprisePetrochemicalSteamCrackerCokingMpcHazopTarAgentSelection(unittest.TestCase):
    def test_tier4_decoking_trips_yield_loss_and_zero_tube_rupture(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["premature_decoking_and_furnace_trip_events_per_year"], 0.5)
        self.assertLessEqual(tier4["ethylene_propylene_olefin_yield_loss_pct"], 0.6)
        self.assertEqual(tier4["radiant_coil_carburization_tube_rupture_risk_pct"], 0.0)
        self.assertLessEqual(tier4["turnaround_tar_critical_path_slip_pct"], 1.5)
        self.assertLess(
            tier4["monthly_yield_decoking_and_tar_loss_usd"],
            tier1["monthly_yield_decoking_and_tar_loss_usd"] * 0.10,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
