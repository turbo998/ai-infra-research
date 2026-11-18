from __future__ import annotations

import unittest
from experiments.enterprise_semiconductor_fab_wafer_yield_aoi_virtual_metrology_agent_selection.run import build_payload


class TestEnterpriseSemiconductorFabWaferYieldAoiVirtualMetrologyAgentSelection(unittest.TestCase):
    def test_tier4_rca_speed_chamber_hit_and_virtual_metrology_gate(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["yield_excursion_rca_hours"], 0.5)
        self.assertLessEqual(tier4["aoi_false_scrap_overkill_pct"], 1.0)
        self.assertEqual(tier4["unmeasured_lot_scrap_escape_pct"], 0.0)
        self.assertGreaterEqual(tier4["chamber_root_cause_hit_rate_pct"], 98.0)
        self.assertLess(tier4["monthly_wafer_scrap_and_metrology_cost_usd"], tier1["monthly_wafer_scrap_and_metrology_cost_usd"] * 0.15)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
