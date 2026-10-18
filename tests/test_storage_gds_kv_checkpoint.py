import unittest

from experiments.storage_gds_kv_checkpoint.run import (
    check_outputs,
    generate_results,
    load_input,
)


class StorageGdsKvCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_input()
        self.results = generate_results(self.config)

    def test_outputs_in_sync(self) -> None:
        self.assertTrue(check_outputs(self.results))

    def test_gds_and_3fs_p2p_bypass_root_complex_bottleneck(self) -> None:
        summary = self.results["summary"]
        self.assertAlmostEqual(summary["kv_read_posix_bw_gbps"], 32.0, places=2)
        self.assertAlmostEqual(summary["kv_read_odirect_bw_gbps"], 48.0, places=2)
        self.assertAlmostEqual(summary["kv_read_gds_bw_gbps"], 54.4, places=2)
        self.assertAlmostEqual(summary["kv_read_3fs_shm_bw_gbps"], 48.0, places=2)
        self.assertAlmostEqual(summary["kv_read_3fs_p2p_bw_gbps"], 90.0, places=2)
        self.assertLess(summary["kv_read_100gib_3fs_p2p_p99_sec"], summary["kv_read_100gib_gds_p99_sec"])
        self.assertLess(summary["ckpt_1tib_3fs_p2p_p99_sec"], summary["ckpt_1tib_posix_p99_sec"] * 0.35)


if __name__ == "__main__":
    unittest.main()
