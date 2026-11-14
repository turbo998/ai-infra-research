import unittest
from experiments.cuda_pinned_memory_copy_engine_full_duplex_selection.run import (
    evaluate_pinned_memory_copy_engine_tiers,
)


class TestCudaPinnedMemoryCopyEngineSelection(unittest.TestCase):
    def test_pinned_copy_engine_tiers(self) -> None:
        data = evaluate_pinned_memory_copy_engine_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_pageable_malloc_silent_sync_staging_fallback"]
        t4 = tiers["tier4_numa_hugepage_pinned_pool_plus_dual_copy_engine_streams"]
        self.assertFalse(t1["true_async_overlap"])
        self.assertTrue(t4["true_async_overlap"])
        self.assertGreater(t4["effective_pci_bandwidth_gbs"], 100.0)


if __name__ == "__main__":
    unittest.main()
