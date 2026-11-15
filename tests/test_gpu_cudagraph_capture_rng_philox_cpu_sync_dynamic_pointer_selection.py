import unittest
from experiments.gpu_cudagraph_capture_rng_philox_cpu_sync_dynamic_pointer_selection.run import (
    evaluate_pytorch_cudagraph_pitfall_tiers,
)


class TestPytorchCudagraphPitfallSelection(unittest.TestCase):
    def test_cudagraph_tiers(self) -> None:
        data = evaluate_pytorch_cudagraph_pitfall_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t2 = tiers["tier2_cudagraph_with_python_variable_rebind_and_host_scalar_args"]
        t4 = tiers["tier4_static_buffer_inplace_copy_plus_device_tensor_metadata_and_philox_rng"]
        self.assertEqual(t2["output_correctness_rate_pct"], 0.0)
        self.assertEqual(t4["output_correctness_rate_pct"], 100.0)
        self.assertEqual(t4["rng_entropy_uniqueness_pct"], 100.0)
        self.assertLess(t4["step_latency_ms"], 6.0)


if __name__ == "__main__":
    unittest.main()
