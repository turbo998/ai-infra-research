from __future__ import annotations

import unittest
from experiments.enterprise_llm_router_cascade_matrix_factorization_abstention_selection.run import build_payload


class TestEnterpriseLlmRouterCascadeMatrixFactorizationAbstentionSelection(unittest.TestCase):
    def test_tier4_quality_retention_low_hallucination_and_cost_reduction(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier4["relative_frontier_quality_retention_pct"], 99.0)
        self.assertLessEqual(tier4["router_decision_overhead_latency_ms"], 10.0)
        self.assertLessEqual(tier4["silent_slm_hallucination_under_routing_pct"], 0.5)
        self.assertLessEqual(tier4["frontier_model_call_share_pct"], 25.0)
        self.assertLess(tier4["monthly_llm_inference_spend_usd"], tier1["monthly_llm_inference_spend_usd"] * 0.30)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
