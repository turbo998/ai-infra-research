import unittest
from experiments.ai_cluster_pcie_acs_p2p_iommu_nic_gpu_direct_rdma_selection.run import (
    evaluate_pcie_acs_iommu_gdr_tiers,
)


class TestPcieAcsIommuGdrSelection(unittest.TestCase):
    def test_pcie_acs_tiers(self) -> None:
        data = evaluate_pcie_acs_iommu_gdr_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_bios_acs_enabled_plus_strict_iommu_root_complex_hairpin"]
        t4 = tiers["tier4_iommu_pt_plus_all_pcie_bridge_setpci_acs_zero_and_peermem"]
        self.assertLess(t1["per_nic_gdr_busbw_gbps"], 10.0)
        self.assertEqual(t4["per_nic_gdr_busbw_gbps"], 47.5)
        self.assertEqual(t4["cpu_root_complex_hairpin_traffic_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
