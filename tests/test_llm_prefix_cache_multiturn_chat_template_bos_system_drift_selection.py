import unittest
from experiments.llm_prefix_cache_multiturn_chat_template_bos_system_drift_selection.run import evaluate_prefix_cache_chat_template_drift


class TestPrefixCacheChatTemplateDrift(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_prefix_cache_chat_template_drift()
        self.assertEqual(res["round"], 288)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_canonical_static_first_plus_periodic_k8_eviction_plus_think_boundary_sync"]["kv_cache_hit_rate_pct"],
            90.0,
        )
        self.assertGreater(
            tiers["tier1_dynamic_timestamp_in_system_prompt_plus_sliding_window"]["p95_ttft_ms"],
            tiers["tier4_canonical_static_first_plus_periodic_k8_eviction_plus_think_boundary_sync"]["p95_ttft_ms"] * 9.0,
        )


if __name__ == "__main__":
    unittest.main()
