import unittest
from experiments.ai_cluster_lustre_weka_3fs_gds_kv_cache_ssd_offload_selection.run import (
    evaluate_kv_cache_nvme_gds_3fs_tiers,
)


class TestKvCacheNvmeGds3fsSelection(unittest.TestCase):
    def test_kv_nvme_gds_tiers(self) -> None:
        data = evaluate_kv_cache_nvme_gds_3fs_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_posix_buffered_io_os_page_cache_triple_copy"]
        t4 = tiers["tier4_2mb_superblock_gds_or_3fs_rdma_plus_gpu_scatter_and_layer_pipeline"]
        self.assertLess(t1["effective_ssd_to_gpu_bw_gbps"], 5.0)
        self.assertGreater(t4["effective_ssd_to_gpu_bw_gbps"], 50.0)
        self.assertLess(t4["ttft_64k_prefix_hit_ms"], 350.0)


if __name__ == "__main__":
    unittest.main()
