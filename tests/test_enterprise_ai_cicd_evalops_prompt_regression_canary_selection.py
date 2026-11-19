from __future__ import annotations

import unittest
from experiments.enterprise_ai_cicd_evalops_prompt_regression_canary_selection.run import build_payload


class TestEnterpriseAiCicdEvalopsPromptRegressionCanarySelection(unittest.TestCase):
    def test_tier4_lowest_regression_escape_fast_pr_ci_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["production_regression_escape_rate_pct"], 1.0)
        self.assertLess(tier4["pr_ci_evaluation_pipeline_duration_min"], 5.0)
        self.assertLess(
            tier4["pr_ci_evaluation_pipeline_duration_min"],
            tier2["pr_ci_evaluation_pipeline_duration_min"] * 0.10,
        )
        self.assertLessEqual(tier4["canary_rollback_mttr_min"], 0.5)
        self.assertLess(
            tier4["monthly_eval_compute_and_incident_tco_usd"],
            tier1["monthly_eval_compute_and_incident_tco_usd"] * 0.20,
        )
        self.assertLess(
            tier4["monthly_eval_compute_and_incident_tco_usd"],
            tier3["monthly_eval_compute_and_incident_tco_usd"] * 0.40,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
