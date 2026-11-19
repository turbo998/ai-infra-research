from __future__ import annotations

import unittest
from experiments.enterprise_saas_ai_tiered_rate_limiting_gcra_token_credit_burst_selection.run import build_payload


class TestEnterpriseSaasAiTieredRateLimitingGcraTokenCreditBurstSelection(unittest.TestCase):
    def test_tier4_ttft_false_429_utilization_and_zero_overrun(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["noisy_neighbor_p99_ttft_latency_ms"], 330.0)
        self.assertLessEqual(tier4["false_429_throttle_on_cached_rag_pct"], 0.5)
        self.assertGreaterEqual(tier4["shared_gpu_cluster_utilization_pct"], 80.0)
        self.assertEqual(tier4["cross_gateway_quota_overrun_leakage_pct"], 0.0)
        self.assertLess(tier4["monthly_gpu_cluster_tco_usd"], tier1["monthly_gpu_cluster_tco_usd"] * 0.40)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
