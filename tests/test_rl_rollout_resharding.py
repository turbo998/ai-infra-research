from __future__ import annotations

import unittest

from experiments.rl_rollout_resharding.run import load_input, run_experiment


class RLRolloutReshardingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_input()
        self.results = run_experiment(self.spec)

    def test_weight_transfer_ordering_and_receiver_staging(self) -> None:
        by_id = {row["id"]: row for row in self.results["weight_transfer_comparison"]}
        self.assertLess(
            by_id["colocated_cuda_ipc_bucketed_512mb"]["sync_latency_sec"],
            by_id["colocated_shm_fallback_512mb"]["sync_latency_sec"],
        )
        self.assertLess(
            by_id["disaggregated_delta_sparse_indices"]["sync_latency_sec"],
            by_id["disaggregated_nccl_multi_sender_nvlink"]["sync_latency_sec"],
        )
        self.assertLess(
            by_id["disaggregated_nccl_multi_sender_nvlink"]["sync_latency_sec"],
            by_id["disaggregated_nccl_single_sender"]["sync_latency_sec"],
        )
        # DeltaShardedCheckpointEngine bounds receiver peak staging to 2 * 512 MiB + 16M * 6 B = 1120 MiB
        self.assertAlmostEqual(by_id["disaggregated_delta_sparse_indices"]["receiver_peak_staging_mib"], 1120.0, places=4)

    def test_cumem_sleep_prevents_80gib_oom(self) -> None:
        by_id = {row["id"]: row for row in self.results["cumem_sleep_comparison"]}
        self.assertFalse(by_id["no_sleep_co_resident"]["fits_on_80gib_gpu"])
        self.assertTrue(by_id["sleep_level_1_offload_weights"]["fits_on_80gib_gpu"])
        self.assertTrue(by_id["sleep_level_2_discard_all_direct_refit"]["fits_on_80gib_gpu"])
        self.assertLess(
            by_id["sleep_level_2_discard_all_direct_refit"]["sleep_wake_transition_sec"],
            by_id["sleep_level_1_offload_weights"]["sleep_wake_transition_sec"],
        )

    def test_long_cot_straggler_bubble_reduction(self) -> None:
        by_id = {row["id"]: row for row in self.results["long_cot_straggler_comparison"]}
        self.assertGreater(by_id["sync_barrier_grpo"]["gpu_idle_bubble_fraction"], 0.75)
        self.assertLess(by_id["async_continuous_staleness_bounded"]["gpu_idle_bubble_fraction"], 0.10)
        self.assertGreater(by_id["async_continuous_staleness_bounded"]["speedup_vs_sync_barrier"], 4.0)


if __name__ == "__main__":
    unittest.main()
