from __future__ import annotations

import unittest
from experiments.enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection.run import build_payload


class TestEnterpriseK8sGatewayApiInferenceExtensionLlmDKvEppRoutingSelection(unittest.TestCase):
    def test_tier4_ttft_prefix_hit_and_zero_kv_preemption_drop(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier4 = tiers[3]
        self.assertLessEqual(tier4["p99_ttft_latency_ms"], 360.0)
        self.assertGreaterEqual(tier4["prefix_cache_and_lora_hit_rate_pct"], 88.0)
        self.assertEqual(tier4["kv_cache_preemption_and_oom_drop_pct"], 0.0)
        self.assertLessEqual(tier4["critical_tier_slo_breach_pct"], 0.2)
        self.assertLess(tier4["monthly_gpu_cluster_tco_usd"], tier1["monthly_gpu_cluster_tco_usd"] * 0.40)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
