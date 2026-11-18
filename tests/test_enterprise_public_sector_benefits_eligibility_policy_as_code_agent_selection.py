from __future__ import annotations

import unittest
from experiments.enterprise_public_sector_benefits_eligibility_policy_as_code_agent_selection.run import build_payload


class TestEnterprisePublicSectorBenefitsEligibilityPolicyAsCodeAgentSelection(unittest.TestCase):
    def test_tier4_zero_statutory_math_error_and_due_process(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["improper_payment_and_wrongful_denial_pct"], 0.5)
        self.assertEqual(tier4["statutory_deduction_math_error_pct"], 0.0)
        self.assertLessEqual(tier4["due_process_appeal_reversal_rate_pct"], 1.0)
        self.assertLessEqual(tier4["application_determination_lag_days"], 0.2)
        self.assertLess(tier4["monthly_improper_payment_and_admin_cost_usd"], tier1["monthly_improper_payment_and_admin_cost_usd"] * 0.12)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
