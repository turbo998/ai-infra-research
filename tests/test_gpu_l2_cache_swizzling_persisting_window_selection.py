import unittest
from experiments.gpu_l2_cache_swizzling_persisting_window_selection.run import (
    evaluate_l2_swizzling_persisting_tiers,
)


class TestGpuL2CacheSwizzlingPersistingSelection(unittest.TestCase):
    def test_l2_swizzle_tiers(self) -> None:
        data = evaluate_l2_swizzling_persisting_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_row_major_cta_order_l2_thrashing"]
        t3 = tiers["tier3_2d_grouped_cta_tile_swizzling_group_m_8"]
        t4 = tiers["tier4_swizzle_plus_l2_persisting_window_and_tma_multicast"]
        self.assertLess(t1["l2_hit_rate_pct"], 30.0)
        self.assertGreater(t3["l2_hit_rate_pct"], 80.0)
        self.assertGreaterEqual(t4["sustained_tflops"], 790.0)


if __name__ == "__main__":
    unittest.main()
