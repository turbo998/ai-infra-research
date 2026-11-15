import unittest
from experiments.pytorch_compile_inductor_cudagraph_trees_dynamic_shape_selection.run import (
    evaluate_torch_compile_cudagraph_trees_tiers,
)


class TestPyTorchCompileCudagraphTreesSelection(unittest.TestCase):
    def test_torch_compile_tiers(self) -> None:
        data = evaluate_torch_compile_cudagraph_trees_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_full_model_compile_static_guard_recompilation_storm"]
        t2 = tiers["tier2_reduce_overhead_cudagraph_trees_multi_bucket_hbm_explosion"]
        t4 = tiers["tier4_piecewise_compile_shared_mempool_cudagraph_buckets"]
        self.assertEqual(t1["runtime_recompilations"], 8)
        self.assertGreater(t2["cudagraph_pool_overhead_gb"], 15.0)
        self.assertEqual(t4["runtime_recompilations"], 0)
        self.assertLess(t4["p99_decode_tpot_ms"], 7.0)


if __name__ == "__main__":
    unittest.main()
