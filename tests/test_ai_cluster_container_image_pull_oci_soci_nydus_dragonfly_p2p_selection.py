import unittest
from experiments.ai_cluster_container_image_pull_oci_soci_nydus_dragonfly_p2p_selection.run import (
    evaluate_container_p2p_nydus_squashfs_tiers,
)


class TestContainerP2pNydusSquashfsSelection(unittest.TestCase):
    def test_container_p2p_nydus_tiers(self) -> None:
        data = evaluate_container_p2p_nydus_squashfs_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_central_harbor_oci_targz_pull_plus_shared_fs_conda_import"]
        t4 = tiers["tier4_nydus_overlaybd_on_demand_p2p_plus_local_nvme_squashfs_loop_mount"]
        self.assertGreater(t1["cold_start_to_step0_sec"], 1600.0)
        self.assertLess(t4["cold_start_to_step0_sec"], 15.0)
        self.assertLess(t4["shared_mds_stat_rpcs_millions"], 0.01)


if __name__ == "__main__":
    unittest.main()
