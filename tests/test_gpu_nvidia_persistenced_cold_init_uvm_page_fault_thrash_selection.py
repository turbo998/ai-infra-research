import unittest
from experiments.gpu_nvidia_persistenced_cold_init_uvm_page_fault_thrash_selection.run import evaluate_persistenced_and_uvm_thrash


class TestPersistencedAndUvmThrash(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_persistenced_and_uvm_thrash()
        self.assertEqual(res["round"], 315)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_persistenced_hot_daemon_plus_explicit_pinned_dma_double_buffer"]["cuda_init_cold_start_sec"],
            0.2,
        )
        self.assertGreater(
            tiers["tier4_persistenced_hot_daemon_plus_explicit_pinned_dma_double_buffer"]["oversubscribed_pcie_bw_gbps"],
            50.0,
        )


if __name__ == "__main__":
    unittest.main()
