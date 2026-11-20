from __future__ import annotations

import unittest
from experiments.enterprise_ai_finops_showback_chargeback_token_budget_governance_selection.run import build_payload


class TestEnterpriseAiFinopsShowbackChargebackTokenBudgetGovernanceSelection(unittest.TestCase):
    def test_tier4_achieves_highest_attribution_zero_outage_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreater(tier4["cost_attribution_accuracy_pct"], 99.0)
        self.assertGreater(
            tier4["cost_attribution_accuracy_pct"],
            tier3["cost_attribution_accuracy_pct"] + 35.0,
        )
        self.assertEqual(tier4["business_outage_from_quota_exhaustion_pct"], 0.0)
        self.assertLess(tier4["monthly_budget_overrun_rate_pct"], 1.0)
        self.assertGreater(
            tier4["shared_ptu_and_gpu_utilization_pct"],
            tier2["shared_ptu_and_gpu_utilization_pct"] + 45.0,
        )
        self.assertLess(
            tier4["monthly_100b_token_enterprise_ai_bill_usd"],
            tier1["monthly_100b_token_enterprise_ai_bill_usd"] * 0.45,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
