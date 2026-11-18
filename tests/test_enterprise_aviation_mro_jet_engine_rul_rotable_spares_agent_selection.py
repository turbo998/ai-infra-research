from __future__ import annotations

import unittest
from experiments.enterprise_aviation_mro_jet_engine_rul_rotable_spares_agent_selection.run import build_payload


class TestEnterpriseAviationMroJetEngineRulRotableSparesAgentSelection(unittest.TestCase):
    def test_tier4_aog_reduction_zero_wrong_part_and_faa_gate(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["unscheduled_aog_events_per_month"], 1.0)
        self.assertEqual(tier4["ipc_effectivity_wrong_part_rate_pct"], 0.0)
        self.assertLessEqual(tier4["premature_green_time_wasted_cycles_pct"], 2.0)
        self.assertEqual(tier4["faa_8130_airworthiness_audit_escape_pct"], 0.0)
        self.assertLess(tier4["monthly_aog_charter_and_spares_cost_usd"], tier1["monthly_aog_charter_and_spares_cost_usd"] * 0.15)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
