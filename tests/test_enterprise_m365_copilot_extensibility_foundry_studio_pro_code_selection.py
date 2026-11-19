from __future__ import annotations

import unittest
from experiments.enterprise_m365_copilot_extensibility_foundry_studio_pro_code_selection.run import build_payload


class TestEnterpriseM365CopilotExtensibilityFoundryStudioProCodeSelection(unittest.TestCase):
    def test_tier4_highest_adoption_fastest_delivery_and_lowest_portfolio_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertGreater(tier4["employee_daily_adoption_and_retention_pct"], 94.0)
        self.assertEqual(tier4["m365_graph_and_sharepoint_acl_fidelity_pct"], 100.0)
        self.assertEqual(tier4["custom_model_and_complex_workflow_flexibility_pct"], 100.0)
        self.assertGreater(
            tier4["custom_model_and_complex_workflow_flexibility_pct"],
            tier2["custom_model_and_complex_workflow_flexibility_pct"] + 50.0,
        )
        self.assertLess(tier4["average_use_case_time_to_production_weeks"], 3.0)
        self.assertLess(
            tier4["monthly_30_use_case_build_and_run_tco_usd"],
            tier1["monthly_30_use_case_build_and_run_tco_usd"] * 0.25,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
