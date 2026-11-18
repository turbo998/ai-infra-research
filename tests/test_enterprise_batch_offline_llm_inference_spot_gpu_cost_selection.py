import unittest
from experiments.enterprise_batch_offline_llm_inference_spot_gpu_cost_selection.run import build_payload


class TestEnterpriseBatchOfflineLlmInferenceSpotGpuCostSelection(unittest.TestCase):
    def test_tier4_minimizes_unit_token_cost_with_zero_sla_misses(self) -> None:
        data = build_payload()
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_synchronous_realtime_paygo_api_loop"]
        t3 = tiers["tier3_managed_cloud_async_batch_api_50pct_off"]
        t4 = tiers["tier4_offpeak_ptu_backfill_plus_spot_k8s_ray_vllm_prefix_sorted"]

        self.assertLessEqual(t4["effective_cost_per_million_tokens_usd"], 0.50)
        self.assertEqual(t4["batch_job_sla_miss_or_429_failure_rate_pct"], 0.0)
        self.assertGreater(t4["gpu_prefix_cache_hit_rate_pct"], 90.0)
        self.assertLess(t4["monthly_batch_inference_cost_usd"], t3["monthly_batch_inference_cost_usd"] * 0.35)
        self.assertLess(t4["monthly_batch_inference_cost_usd"], t1["monthly_batch_inference_cost_usd"] * 0.2)

    def test_provenance_explicit(self) -> None:
        data = build_payload()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")


if __name__ == "__main__":
    unittest.main()
