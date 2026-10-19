import unittest

from experiments.sglang_hiradix_cache_controller.run import build_summary, run_check


class TestSglangHiRadixCacheController(unittest.TestCase):
    def test_mla_host_dedup_capacity_and_bandwidth(self) -> None:
        summary = build_summary()
        run_check(summary)
        mla = summary["mla_host_dedup"]
        self.assertEqual(mla["effective_capacity_multiplier"], 8.0)
        self.assertLess(mla["prefix_32k_load_with_dedup_ms"], mla["prefix_32k_load_no_dedup_ms"])

    def test_trailing_pages_intersection_and_hol_cancel(self) -> None:
        summary = build_summary()
        sp = summary["storage_prefetch_and_hit_policies"]
        self.assertGreater(sp["hol_wait_reduction_pct"], 90.0)
        self.assertFalse(sp["trailing_pages_intersection"]["naive_min_is_valid_on_all_ranks"])
        self.assertTrue(sp["trailing_pages_intersection"]["safe_is_valid_on_all_ranks"])
        self.assertEqual(sp["trailing_pages_intersection"]["safe_restorable_prefix_page"], 32)


if __name__ == "__main__":
    unittest.main()
