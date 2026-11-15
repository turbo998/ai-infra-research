import unittest
from experiments.llm_pytorch_caching_allocator_fragmentation_max_split_vmm_selection.run import (
    evaluate_pytorch_cuda_alloc_conf_tiers,
)


class TestPyTorchCudaAllocConfSelection(unittest.TestCase):
    def test_cuda_alloc_conf_tiers(self) -> None:
        data = evaluate_pytorch_cuda_alloc_conf_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_default_cudamalloc_allocator_14gb_swiss_cheese_oom"]
        t2 = tiers["tier2_hot_loop_torch_cuda_empty_cache_performance_suicide"]
        t4 = tiers["tier4_expandable_segments_true_cumemmap_virtual_stitching"]
        self.assertGreater(t1["unusable_fragmented_reserved_hbm_gb"], 14.0)
        self.assertGreater(t2["allocator_stall_overhead_ms_per_step"], 300.0)
        self.assertLess(t4["unusable_fragmented_reserved_hbm_gb"], 0.5)
        self.assertLess(t4["allocator_stall_overhead_ms_per_step"], 1.0)


if __name__ == "__main__":
    unittest.main()
