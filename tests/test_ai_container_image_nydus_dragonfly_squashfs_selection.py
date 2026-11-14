import unittest
from experiments.ai_container_image_nydus_dragonfly_squashfs_selection.run import (
    evaluate_container_and_import_tiers,
)


class TestContainerNydusSquashfsSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        data = evaluate_container_and_import_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        self.assertGreater(data["workload"]["cold_start_speedup_x"], 150.0)
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_central_harbor_gzip_plus_shared_nfs_conda"]
        t4 = tiers["tier4_nydus_erofs_p2p_plus_squashfs_loopback_conda"]
        self.assertEqual(t1["total_job_cold_start_s"], 580.0)
        self.assertLessEqual(t4["total_job_cold_start_s"], 4.0)
        self.assertLess(t4["shared_mds_metadata_rpcs_millions"], 0.01)


if __name__ == "__main__":
    unittest.main()
