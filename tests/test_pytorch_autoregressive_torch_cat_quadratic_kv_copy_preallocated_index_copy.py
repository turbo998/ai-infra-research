import unittest
from experiments.pytorch_autoregressive_torch_cat_quadratic_kv_copy_preallocated_index_copy_selection.run import evaluate


class TestPyTorchAutoregressiveTorchCatQuadraticKVCopyPreallocatedIndexCopySelection(unittest.TestCase):
    def test_evaluate_metrics(self) -> None:
        res = evaluate()
        t1 = res["tiers"]["tier1_per_step_torch_cat"]
        t2 = res["tiers"]["tier2_chunk64_torch_cat"]
        t3 = res["tiers"]["tier3_preallocated_python_int_slice"]
        t4 = res["tiers"]["tier4_preallocated_tensor_index_copy"]

        self.assertAlmostEqual(t1["total_copy_traffic_gb"], 536.87, places=2)
        self.assertEqual(t1["dynamic_allocations"], 4096)
        self.assertAlmostEqual(t1["cumulative_copy_latency_ms"], 168.40, places=2)
        self.assertAlmostEqual(t2["total_copy_traffic_gb"], 8.39, places=2)
        self.assertFalse(t3["cudagraph_safe"])
        self.assertAlmostEqual(t4["total_copy_traffic_gb"], 0.26, places=2)
        self.assertAlmostEqual(t4["cumulative_copy_latency_ms"], 1.12, places=2)
        self.assertTrue(t4["cudagraph_safe"])
        self.assertAlmostEqual(t4["speedup_vs_tier1_x"], 150.4, places=1)


if __name__ == "__main__":
    unittest.main()
