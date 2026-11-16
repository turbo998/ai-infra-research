import unittest
from experiments.pytorch_transpose_contiguous_copy_explosion_stride_aware_layout_selection.run import evaluate


class TestPyTorchTransposeContiguousCopyExplosionStrideAwareLayoutSelection(unittest.TestCase):
    def test_evaluate_metrics(self) -> None:
        res = evaluate()
        t1 = res["tiers"]["tier1_explicit_transpose_contiguous_cat_rope"]
        t2 = res["tiers"]["tier2_silent_reshape_copy_on_transposed"]
        t3 = res["tiers"]["tier3_inner_dim_stride_broken_transpose"]
        t4 = res["tiers"]["tier4_native_bshd_stride_aware_inplace_rope"]

        self.assertAlmostEqual(t1["copy_traffic_gb_per_layer"], 24.00, places=2)
        self.assertAlmostEqual(t1["layout_rope_latency_ms"], 7.16, places=2)
        self.assertAlmostEqual(t2["copy_traffic_gb_per_layer"], 12.00, places=2)
        self.assertAlmostEqual(t3["copy_traffic_gb_per_layer"], 6.00, places=2)
        self.assertAlmostEqual(t4["copy_traffic_gb_per_layer"], 0.00, places=2)
        self.assertAlmostEqual(t4["layout_rope_latency_ms"], 0.18, places=2)
        self.assertAlmostEqual(t4["speedup_vs_tier1_x"], 39.8, places=1)
        self.assertAlmostEqual(t4["saved_hbm_per_layer_gb"], 4.00, places=2)


if __name__ == "__main__":
    unittest.main()
