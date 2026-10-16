"""Unit tests for the distributed checkpoint deduplication, LPT bucketing, and GDS/POSIX recovery experiment."""

import json
from pathlib import Path
import unittest

from experiments.checkpoint_recovery.run import (
    HERE,
    calculate,
    evaluate_checkpoint_case,
    main,
)


class CheckpointRecoveryTest(unittest.TestCase):
    def test_fixture_check_passes(self) -> None:
        self.assertEqual(main(["--check"]), 0)

    def test_deduplication_and_loader_modes(self) -> None:
        fixture = json.loads((HERE / "input.json").read_text(encoding="utf-8"))
        output = calculate(fixture)
        by_id = {item["result"]["case_id"]: item["result"] for item in output["cases"]}

        multi_sync = by_id["dcp_sync_multi_thread_serial_fallback"]
        self.assertEqual(multi_sync["loader_mode"], "serial_cpu_loader_multi_thread_fallback")
        self.assertEqual(multi_sync["dedup_saved_bytes"], 15032385536)
        self.assertLess(
            multi_sync["max_rank_write_bytes_balanced"],
            multi_sync["max_rank_write_bytes_lowest_rank"],
        )

        single_overlap = by_id["dcp_sync_single_thread_overlapping_loader"]
        self.assertEqual(single_overlap["loader_mode"], "overlapping_cpu_loader_single_thread")
        self.assertLess(
            single_overlap["step_blocking_time_seconds"],
            multi_sync["step_blocking_time_seconds"],
        )

        async_staged = by_id["dcp_async_save_pinned_cpu_stager"]
        self.assertEqual(async_staged["loader_mode"], "async_blocking_stager_then_cpu_threads")
        self.assertLess(async_staged["step_blocking_time_seconds"], 1.0)
        self.assertGreater(
            async_staged["steady_state_goodput_fraction"],
            multi_sync["steady_state_goodput_fraction"],
        )

        gds_direct = by_id["cufile_gds_direct_dma_aligned"]
        self.assertTrue(gds_direct["effective_gds_direct_dma"])
        self.assertEqual(gds_direct["host_memory_bounce_bytes_per_rank"], 0)

        gds_fallback = by_id["cufile_gds_unaligned_compat_fallback"]
        self.assertTrue(gds_fallback["gds_compatibility_fallback"])
        self.assertFalse(gds_fallback["effective_gds_direct_dma"])
        self.assertGreater(gds_fallback["host_memory_bounce_bytes_per_rank"], 0)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_checkpoint_case(
                case_id="bad_threads",
                world_size=4,
                sharded_tensor_bytes_per_rank=[1024],
                replicated_tensor_bytes=[512],
                writer_thread_count=0,
                pcie_d2h_bytes_per_sec_per_rank=1000,
                host_ram_Copy_bytes_per_sec_per_rank=1000,
                storage_write_bytes_per_sec_per_rank=1000,
                cluster_aggregate_storage_bytes_per_sec=4000,
                metadata_fsync_barrier_seconds=0.1,
                mtbf_hours=10.0,
            )


if __name__ == "__main__":
    unittest.main()
