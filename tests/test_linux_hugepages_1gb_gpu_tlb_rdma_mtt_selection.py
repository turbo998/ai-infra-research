import unittest
from experiments.linux_hugepages_1gb_gpu_tlb_rdma_mtt_selection.run import (
    evaluate_hugepage_tlb_tiers,
)


class TestHugepageTlbMttSelection(unittest.TestCase):
    def test_tiers_and_pte_count(self) -> None:
        data = evaluate_hugepage_tlb_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertEqual(data["workload"]["pages_4kb"], 134217728)
        self.assertEqual(data["workload"]["pages_1gb"], 512)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_standard_4kb_os_pages"]
        t4 = tiers["tier4_boot_1gb_hugepages_plus_grace_64kb_ats"]
        self.assertLess(t1["random_kv_dma_bandwidth_gbs"], 22.0)
        self.assertGreaterEqual(t4["random_kv_dma_bandwidth_gbs"], 54.0)
        self.assertLessEqual(t4["mr_pin_registration_time_s"], 0.1)


if __name__ == "__main__":
    unittest.main()
