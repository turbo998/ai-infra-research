import unittest

from experiments.async_dcp_checkpoint_stager.run import build_summary, run_check


class TestAsyncDcpCheckpointStager(unittest.TestCase):
    def test_dcp_staging_strategies_and_goodput(self) -> None:
        summary = build_summary()
        run_check(summary)
        hl = summary["headline_metrics"]
        self.assertGreater(hl["stall_reduction_vs_sync_pct"], 99.9)
        self.assertLess(hl["cached_default_stager_exposed_stall_s"], 0.1)
        self.assertGreater(hl["cached_goodput_pct"], 99.98)

    def test_replication_stager_half_world_pairing(self) -> None:
        summary = build_summary()
        repl = summary["replication_topology"]
        self.assertTrue(repl["deadlock_free_schedule"])
        self.assertTrue(repl["disjoint_rack_failure_domain"])
        self.assertEqual(repl["unique_bidirectional_pairs"], 8)


if __name__ == "__main__":
    unittest.main()
