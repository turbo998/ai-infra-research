import unittest
from experiments.pytorch_dataloader_fork_cow_memory_leak_shm_serialization_selection.run import (
    evaluate_dataloader_fork_cow_shm_tiers,
)


class TestPyTorchDataLoaderForkCowShmSelection(unittest.TestCase):
    def test_dataloader_cow_tiers(self) -> None:
        data = evaluate_dataloader_fork_cow_shm_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_python_list_dict_fork_cow_refcount_64x_ram_explosion"]
        t4 = tiers["tier4_mmap_binary_index_plus_persistent_workers_and_64gb_shm"]
        self.assertEqual(t1["steady_state_node_host_ram_gb"], 768.0)
        self.assertLess(t4["steady_state_node_host_ram_gb"], 15.0)
        self.assertEqual(t4["shm_sigbus_or_oom_crashes_per_run"], 0)
        self.assertEqual(t4["cow_page_duplication_factor_x"], 1.0)


if __name__ == "__main__":
    unittest.main()
