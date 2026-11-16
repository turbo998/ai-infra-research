import unittest
from experiments.llm_prefix_cache_timing_side_channel_tenant_salted_sha256_eviction_selection.run import (
    evaluate_prefix_cache_security_tiers,
)


class TestPrefixCacheSecuritySelection(unittest.TestCase):
    def test_tiers_and_security(self) -> None:
        res = evaluate_prefix_cache_security_tiers()
        self.assertAlmostEqual(res["timing_oracle_ratio_x"], 9.07, places=2)
        self.assertAlmostEqual(res["unsalted_prompt_stealing_pct"], 98.4, places=1)
        self.assertAlmostEqual(res["salted_prompt_stealing_pct"], 0.0, places=1)
        self.assertAlmostEqual(res["hit_rate_gain_pt"], 43.4, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["cross_tenant_leak_accuracy_pct"], 98.4)
        self.assertEqual(t4["cross_tenant_leak_accuracy_pct"], 0.0)
        self.assertEqual(t4["sustained_cache_hit_rate_pct"], 84.6)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
