from __future__ import annotations

import unittest
from experiments.enterprise_ai_observability_otel_tracing_finops_chargeback_eval_selection.run import build_payload


class TestEnterpriseAiObservabilityOtelTracingFinopsChargebackEvalSelection(unittest.TestCase):
    def test_tier4_fastest_mttr_highest_finops_accuracy_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["multi_step_agent_root_cause_mttr_min"], 10.0)
        self.assertGreater(tier4["business_unit_finops_cost_attribution_accuracy_pct"], 99.0)
        self.assertGreater(
            tier4["silent_quality_drift_detection_recall_pct"],
            tier2["silent_quality_drift_detection_recall_pct"] + 75.0,
        )
        self.assertEqual(tier4["observability_pii_leak_and_vendor_lock_risk_pct"], 0.0)
        self.assertLess(
            tier4["monthly_observability_and_unallocated_waste_tco_usd"],
            tier3["monthly_observability_and_unallocated_waste_tco_usd"] * 0.25,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
