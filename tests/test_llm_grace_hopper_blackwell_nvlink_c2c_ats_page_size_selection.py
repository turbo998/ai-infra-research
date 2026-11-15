import unittest
from experiments.llm_grace_hopper_blackwell_nvlink_c2c_ats_page_size_selection.run import (
    evaluate_gb200_nvlink_c2c_ats_page_tiers,
)


class TestGb200NvlinkC2cAtsPageSelection(unittest.TestCase):
    def test_nvlink_c2c_tiers(self) -> None:
        data = evaluate_gb200_nvlink_c2c_ats_page_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_arm64_4kb_base_page_kernel_without_thp_ats_tlb_thrashing"]
        t4 = tiers["tier4_arm64_64kb_kernel_plus_512mb_thp_and_zero_copy_coherent_c2c"]
        self.assertEqual(t1["tlb_entries_for_128gb"], 33554432)
        self.assertEqual(t4["tlb_entries_for_128gb"], 256)
        self.assertGreater(t4["effective_c2c_unidir_bw_gbps"], 420.0)


if __name__ == "__main__":
    unittest.main()
