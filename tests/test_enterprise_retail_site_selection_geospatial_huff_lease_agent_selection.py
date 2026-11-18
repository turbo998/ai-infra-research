from __future__ import annotations

import unittest
from experiments.enterprise_retail_site_selection_geospatial_huff_lease_agent_selection.run import build_payload


class TestEnterpriseRetailSiteSelectionGeospatialHuffLeaseAgentSelection(unittest.TestCase):
    def test_tier4_revenue_mape_cannibalization_and_lease_gate(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["first_year_revenue_mape_pct"], 7.0)
        self.assertLessEqual(tier4["sister_store_cannibalization_error_pct"], 3.0)
        self.assertEqual(tier4["lease_trap_escape_rate_pct"], 0.0)
        self.assertLess(tier4["monthly_underwriting_bad_capex_cost_usd"], tier1["monthly_underwriting_bad_capex_cost_usd"] * 0.15)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
