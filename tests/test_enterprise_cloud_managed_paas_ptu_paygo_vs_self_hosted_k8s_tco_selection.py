from __future__ import annotations

import unittest
from experiments.enterprise_cloud_managed_paas_ptu_paygo_vs_self_hosted_k8s_tco_selection.run import build_payload


class TestEnterpriseCloudManagedPaasPtuPaygoVsSelfHostedK8sTcoSelection(unittest.TestCase):
    def test_tier4_lowest_blended_cost_zero_throttling_and_lowest_full_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["effective_blended_cost_per_1m_tokens_usd"], 0.80)
        self.assertEqual(tier4["peak_hour_http_429_throttling_rate_pct"], 0.0)
        self.assertLess(tier4["p99_ttft_sla_guarantee_ms"], 450.0)
        self.assertLess(
            tier4["monthly_total_compute_and_sre_fte_tco_usd"],
            tier1["monthly_total_compute_and_sre_fte_tco_usd"] * 0.35,
        )
        self.assertLess(
            tier4["monthly_total_compute_and_sre_fte_tco_usd"],
            tier2["monthly_total_compute_and_sre_fte_tco_usd"] * 0.45,
        )
        self.assertLess(
            tier4["monthly_total_compute_and_sre_fte_tco_usd"],
            tier3["monthly_total_compute_and_sre_fte_tco_usd"] * 0.60,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
