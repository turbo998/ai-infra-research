import unittest
from experiments.pytorch_dist_all_gather_object_pickle_cuda_sync_stall_selection.run import evaluate


class TestPyTorchDistAllGatherObjectPickleCudaSyncStallSelection(unittest.TestCase):
    def test_evaluate_metrics(self) -> None:
        res = evaluate()
        t1 = res["tiers"]["tier1_nccl_all_gather_object"]
        t2 = res["tiers"]["tier2_cuda_tensor_in_pickled_dict"]
        t3 = res["tiers"]["tier3_gloo_cpu_all_gather_object"]
        t4 = res["tiers"]["tier4_fixed_schema_all_gather_into_tensor"]

        self.assertAlmostEqual(t1["step_stall_ms"], 14.80, places=2)
        self.assertEqual(t1["gpu_stream_syncs_per_call"], 2)
        self.assertAlmostEqual(t2["step_stall_ms"], 185.00, places=2)
        self.assertAlmostEqual(t3["step_stall_ms"], 3.40, places=2)
        self.assertAlmostEqual(t4["step_stall_ms"], 0.08, places=2)
        self.assertEqual(t4["gpu_stream_syncs_per_call"], 0)
        self.assertAlmostEqual(t4["speedup_vs_tier1_x"], 185.0, places=1)


if __name__ == "__main__":
    unittest.main()
