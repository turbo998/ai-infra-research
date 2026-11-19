from __future__ import annotations

import unittest
from experiments.enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection.run import build_payload


class TestEnterpriseInternalMaasPlatformMultiTenantGpuSharingSelection(unittest.TestCase):
    def test_tier4_highest_utilization_zero_interference_instant_onboarding_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertGreater(tier4["cluster_wide_gpu_utilization_pct"], 85.0)
        self.assertEqual(tier4["noisy_neighbor_oom_or_latency_interference_pct"], 0.0)
        self.assertGreater(tier2["noisy_neighbor_oom_or_latency_interference_pct"], 30.0)
        self.assertLess(tier4["new_project_onboarding_lead_time_days"], 0.05)
        self.assertLessEqual(tier4["required_h100_gpu_equivalents"], 24)
        self.assertLess(
            tier4["monthly_internal_maas_platform_tco_usd"],
            tier1["monthly_internal_maas_platform_tco_usd"] * 0.30,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
