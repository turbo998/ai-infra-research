from __future__ import annotations

import unittest
from experiments.enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection.run import build_payload


class TestEnterpriseMultiCloudSpotGpuSkyDwsKueueFallbackServingSelection(unittest.TestCase):
    def test_tier4_stockout_zero_drop_and_zero_partial_gang_waste(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["gpu_allocation_stockout_failure_rate_pct"], 0.2)
        self.assertEqual(tier4["spot_preemption_dropped_inflight_requests_pct"], 0.0)
        self.assertEqual(tier4["partial_gang_allocation_idle_gpu_waste_pct"], 0.0)
        self.assertLessEqual(tier4["failover_and_replacement_recovery_seconds"], 15.0)
        self.assertLess(tier4["monthly_multi_cloud_gpu_compute_tco_usd"], tier1["monthly_multi_cloud_gpu_compute_tco_usd"] * 0.40)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
