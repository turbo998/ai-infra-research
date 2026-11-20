from __future__ import annotations

import unittest
from experiments.enterprise_ai_sandbox_synthetic_test_data_anonymization_staging_selection.run import build_payload


class TestEnterpriseAiSandboxSyntheticTestDataAnonymizationStagingSelection(unittest.TestCase):
    def test_tier4_achieves_zero_pii_risk_high_join_fidelity_and_fast_ci(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertEqual(tier4["non_prod_pii_compliance_breach_risk_pct"], 0.0)
        self.assertGreater(tier4["referential_join_and_schema_fidelity_pct"], 99.0)
        self.assertGreater(
            tier4["referential_join_and_schema_fidelity_pct"],
            tier2["referential_join_and_schema_fidelity_pct"] + 50.0,
        )
        self.assertLess(
            tier4["ci_integration_test_suite_duration_minutes"],
            tier1["ci_integration_test_suite_duration_minutes"] * 0.10,
        )
        self.assertLess(
            tier4["monthly_non_prod_ai_sandbox_tco_usd"],
            tier2["monthly_non_prod_ai_sandbox_tco_usd"] * 0.20,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
