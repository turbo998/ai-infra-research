import unittest
from experiments.gpudirect_rdma_pcie_acs_iommu_dmabuf_selection.run import (
    evaluate_gpudirect_pcie_acs_tiers,
)


class TestGpuDirectPcieAcsSelection(unittest.TestCase):
    def test_tiers_and_busbw(self) -> None:
        data = evaluate_gpudirect_pcie_acs_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_host_bounce_buffer_no_gdr"]
        t2 = tiers["tier2_gdr_with_bios_acs_enabled_and_iommu_strict"]
        t4 = tiers["tier4_pix_acs_off_iommu_pt_kernel_dmabuf"]
        self.assertLess(t1["nccl_busbw_gbs"], 12.0)
        self.assertLess(t2["nccl_busbw_gbs"], 20.0)
        self.assertGreaterEqual(t4["nccl_busbw_gbs"], 48.0)


if __name__ == "__main__":
    unittest.main()
