import unittest
from experiments.llm_checkpoint_dcp_inode_storm_odirect_async_pinned_selection.run import (
    evaluate_checkpoint_dcp_inode_odirect_tiers,
)


class TestCheckpointDcpInodeOdirectSelection(unittest.TestCase):
    def test_checkpoint_dcp_tiers(self) -> None:
        data = evaluate_checkpoint_dcp_inode_odirect_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_per_tensor_file_mds_inode_lock_storm"]
        t4 = tiers["tier4_persistent_pinned_async_stager_plus_odirect_two_phase_commit"]
        self.assertGreater(t1["gpu_training_stall_sec"], 400.0)
        self.assertLess(t4["gpu_training_stall_sec"], 0.05)
        self.assertEqual(t4["host_page_cache_bloat_gb_per_node"], 0.0)


if __name__ == "__main__":
    unittest.main()
