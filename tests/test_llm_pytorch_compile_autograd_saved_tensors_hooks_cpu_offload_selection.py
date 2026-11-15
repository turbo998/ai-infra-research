import unittest
from experiments.llm_pytorch_compile_autograd_saved_tensors_hooks_cpu_offload_selection.run import (
    evaluate_saved_tensors_hooks_offload_tiers,
)


class TestSavedTensorsHooksOffloadSelection(unittest.TestCase):
    def test_saved_tensors_hooks_tiers(self) -> None:
        data = evaluate_saved_tensors_hooks_offload_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_naive_save_on_cpu_blocks_in_unpack_and_offloads_weights"]
        t4 = tiers["tier4_checkpoint_boundary_offload_plus_layer_ahead_h2d_prefetch"]
        self.assertEqual(t1["pcie_transfer_overlap_pct"], 0.0)
        self.assertGreater(t4["pcie_transfer_overlap_pct"], 95.0)
        self.assertLess(t4["step_latency_ms"], 1150.0)


if __name__ == "__main__":
    unittest.main()
