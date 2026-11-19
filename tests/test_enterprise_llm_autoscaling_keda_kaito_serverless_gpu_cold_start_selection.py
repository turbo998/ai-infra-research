from __future__ import annotations

import unittest
from experiments.enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection.run import build_payload


class TestEnterpriseLlmAutoscalingKedaKaitoServerlessGpuColdStartSelection(unittest.TestCase):
    def test_tier4_fast_cold_start_zero_drops_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["scale_out_pod_ready_latency_sec"], 40.0)
        self.assertEqual(tier4["traffic_spike_5x_request_drop_or_timeout_pct"], 0.0)
        self.assertGreater(tier4["average_24x7_gpu_fleet_utilization_pct"], 80.0)
        self.assertLess(tier4["p99_ttft_during_morning_surge_sec"], tier2["p99_ttft_during_morning_surge_sec"] * 0.1)
        self.assertLess(
            tier4["monthly_inference_compute_tco_usd"],
            tier1["monthly_inference_compute_tco_usd"] * 0.45,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
