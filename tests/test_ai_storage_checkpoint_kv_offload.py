from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.ai_storage_checkpoint_kv_offload.run import RESULTS_PATH, build_results


class TestAiStorageCheckpointKvOffload(unittest.TestCase):
    def test_results_match_deterministic_model(self) -> None:
        computed = build_results()
        self.assertTrue(RESULTS_PATH.exists(), "results.json must exist")
        on_disk = json.loads(Path(RESULTS_PATH).read_text(encoding="utf-8"))
        self.assertEqual(computed, on_disk)

    def test_arithmetic_and_provenance_invariants(self) -> None:
        data = build_results()
        self.assertFalse(data["provenance_details"]["independently_reproduced"])
        self.assertFalse(data["provenance_details"]["paid_cloud_gpu_launched"])
        self.assertEqual(data["provenance_details"]["live_512gpu_cluster_measurement"], "not_measured")

        # Verify Checkpoint arithmetic: 9000 GB / 2880 GB/s = 3.125 s, 225.0 / 3.125 = 72.0x
        modes = {m["mode"]: m for m in data["checkpoint_scenario"]["modes"]}
        self.assertAlmostEqual(modes["sync_posix_lustre"]["gpu_stall_s"], 225.0)
        self.assertAlmostEqual(modes["sync_gds_rdma_fs"]["gpu_stall_s"], 25.0)
        self.assertAlmostEqual(modes["two_phase_async_dram_nvme"]["gpu_stall_s"], 3.125)
        self.assertAlmostEqual(modes["two_phase_async_dram_nvme"]["stall_speedup_vs_posix"], 72.0)

        # Verify KV tiering arithmetic: 2.048 GB / 51.2 GB/s = 40.0 ms + 12.0 ms = 52.0 ms -> 780 / 52 = 15.0x
        tiers = {t["tier"]: t for t in data["kv_tiering_scenario"]["tiers"]}
        self.assertAlmostEqual(tiers["host_dram_pinned_pcie5"]["kv_reload_ms"], 40.0)
        self.assertAlmostEqual(tiers["host_dram_pinned_pcie5"]["resume_ttft_ms"], 52.0)
        self.assertAlmostEqual(tiers["host_dram_pinned_pcie5"]["ttft_speedup_vs_recompute"], 15.0)
        self.assertAlmostEqual(tiers["shared_rdma_nvme_gds"]["kv_reload_ms"], 50.0)
        self.assertAlmostEqual(tiers["shared_rdma_nvme_gds"]["resume_ttft_ms"], 62.0)
        self.assertAlmostEqual(tiers["shared_rdma_nvme_gds"]["ttft_speedup_vs_recompute"], 12.58)


if __name__ == "__main__":
    unittest.main()
