import unittest
from experiments.ai_cluster_lustre_beegfs_small_file_webdataset_mds_bottleneck_selection.run import evaluate_pfs_small_file_mds_vs_sharded


class TestPfsSmallFileMdsVsSharded(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_pfs_small_file_mds_vs_sharded()
        self.assertEqual(res["round"], 296)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier4_mosaicml_mds_indexed_shards_plus_lfs_stripe1_plus_local_nvme_cache"]["cluster_read_bandwidth_gbps"],
            tiers["tier1_raw_100m_individual_files_on_lustre_mds_collapse"]["cluster_read_bandwidth_gbps"] * 120.0,
        )
        self.assertLess(
            tiers["tier4_mosaicml_mds_indexed_shards_plus_lfs_stripe1_plus_local_nvme_cache"]["gpu_data_stall_pct"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
