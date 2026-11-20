from __future__ import annotations

import unittest
from experiments.enterprise_llm_version_upgrade_deprecation_migration_shadow_eval_selection.run import build_payload


class TestEnterpriseLlmVersionUpgradeDeprecationMigrationShadowEvalSelection(unittest.TestCase):
    def test_tier4_eliminates_silent_regressions_and_minimizes_migration_cost(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["post_upgrade_silent_regression_rate_pct"], 0.5)
        self.assertLess(
            tier4["post_upgrade_silent_regression_rate_pct"],
            tier1["post_upgrade_silent_regression_rate_pct"] * 0.02,
        )
        self.assertLess(
            tier4["migration_engineering_lead_time_days"],
            tier2["migration_engineering_lead_time_days"] * 0.15,
        )
        self.assertGreater(tier4["automated_prompt_drift_remediation_coverage_pct"], 95.0)
        self.assertLess(
            tier4["per_migration_wave_incident_and_labor_cost_usd"],
            tier3["per_migration_wave_incident_and_labor_cost_usd"] * 0.30,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
