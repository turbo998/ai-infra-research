import unittest
from experiments.tiered_kv_cache_hierarchy import run as exp


class TestTieredKVCacheHierarchy(unittest.TestCase):
    def test_results_check_and_breakeven(self):
        self.assertEqual(exp.main(["--check"]), 0)
        res = exp.generate_results()
        self.assertEqual(res["evidence_status"], "own-offline")
        sm = res["summary_metrics"]
        self.assertAlmostEqual(sm["ddr5_64k_swap_ms"], 24.32, places=2)
        self.assertAlmostEqual(sm["cxl_mooncake_64k_swap_ms"], 30.48, places=2)
        self.assertAlmostEqual(sm["nvme_gds_64k_swap_ms"], 208.29, places=2)
        self.assertEqual(sm["posix_disk_breakeven_tokens"], 6940)
        self.assertAlmostEqual(sm["four_tier_p99_ttft_ms"], 231.99, places=2)
        self.assertGreater(sm["p99_ttft_speedup_vs_hbm_only"], 18.0)
        self.assertAlmostEqual(sm["token_cost_reduction_pct"], 66.6, places=1)


if __name__ == "__main__":
    unittest.main()
