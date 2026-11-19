from __future__ import annotations

import unittest
from experiments.enterprise_ai_caching_prefix_kv_cache_semantic_cache_redis_selection.run import build_payload


class TestEnterpriseAiCachingPrefixKvCacheSemanticCacheRedisSelection(unittest.TestCase):
    def test_tier4_highest_token_savings_zero_collisions_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier4 = tiers[3]
        self.assertGreater(tier4["effective_input_token_cost_reduction_pct"], 75.0)
        self.assertEqual(tier4["semantic_collision_wrong_answer_rate_pct"], 0.0)
        self.assertGreater(tier2["semantic_collision_wrong_answer_rate_pct"], 10.0)
        self.assertEqual(tier4["cross_tenant_acl_cache_poisoning_rate_pct"], 0.0)
        self.assertLess(tier4["p95_ttft_latency_ms"], 200.0)
        self.assertLess(
            tier4["monthly_llm_input_and_cache_tco_usd"],
            tier1["monthly_llm_input_and_cache_tco_usd"] * 0.25,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
