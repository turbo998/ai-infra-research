import unittest
from experiments.gpudirect_storage_gds_cufile_nvme_iouring_selection.run import (
    evaluate_gds_nvme_tiers,
)


class TestGdsNvmeSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        data = evaluate_gds_nvme_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertGreater(data["workload"]["throughput_speedup_x"], 5.0)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_posix_buffered_page_cache_double_copy"]
        t4 = tiers["tier4_gds_cufile_pcie_switch_pix_p2p_zero_copy"]
        self.assertLess(t1["aggregate_nvme_to_gpu_gbs"], 20.0)
        self.assertGreaterEqual(t4["aggregate_nvme_to_gpu_gbs"], 98.0)
        self.assertEqual(t4["host_dram_traffic_multiplier"], 0.0)


if __name__ == "__main__":
    unittest.main()
