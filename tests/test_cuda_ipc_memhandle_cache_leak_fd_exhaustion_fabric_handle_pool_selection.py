import unittest
from experiments.cuda_ipc_memhandle_cache_leak_fd_exhaustion_fabric_handle_pool_selection.run import evaluate_cuda_ipc_handle_pool


class TestCudaIpcHandlePool(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_cuda_ipc_handle_pool()
        self.assertEqual(res["round"], 322)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_vmm_cu_mem_export_shareable_fabric_fd_pool_plus_offset_registry"]["ipc_mapping_overhead_ms_per_step"],
            0.01,
        )
        self.assertTrue(
            tiers["tier4_vmm_cu_mem_export_shareable_fabric_fd_pool_plus_offset_registry"]["supports_vmm_and_gb200_mnnvl"]
        )


if __name__ == "__main__":
    unittest.main()
