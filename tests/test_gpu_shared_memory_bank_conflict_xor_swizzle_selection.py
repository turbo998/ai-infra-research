import unittest
from experiments.gpu_shared_memory_bank_conflict_xor_swizzle_selection.run import (
    evaluate_smem_swizzle_tiers,
)


class TestGpuSmemBankConflictXorSwizzleSelection(unittest.TestCase):
    def test_smem_swizzle_tiers(self) -> None:
        data = evaluate_smem_swizzle_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(data["workload"]["bank_line_bytes"], 128)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_unswizzled_row_major_16way_bank_conflict"]
        t2 = tiers["tier2_legacy_plus8_row_padding_alignment_broken"]
        t4 = tiers["tier4_hardware_tma_wgmma_swizzle_128b_descriptor"]
        self.assertGreater(t1["bank_conflict_way"], 1)
        self.assertGreater(t2["wasted_sram_pct"], 10.0)
        self.assertEqual(t4["wasted_sram_pct"], 0.0)
        self.assertGreaterEqual(t4["effective_sram_bandwidth_tbs"], 18.5)


if __name__ == "__main__":
    unittest.main()
