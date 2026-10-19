"""Unit tests for NCCL PAT (`NCCL_ALGO_PAT`) Parallel Aggregated Trees AllGather & ReduceScatter Simulator."""

from __future__ import annotations

import subprocess
import sys
import unittest

from experiments.nccl_pat_binomial_allgather_rs.run import (
    RESULTS_PATH,
    compute_pat_aggregation_params,
    first_bit_set,
    mirror_invert,
    run_pat_experiment,
)


class TestNcclPatBinomialAllGatherRS(unittest.TestCase):
    def test_mirror_invert_and_first_bit_set(self) -> None:
        # In PatRSAlgorithm (`src/include/collectives.h:485-501`), for max=16:
        # i=0 -> mirrorInvert(0, 16) = 15; i=1 -> 7; i=2 -> 11; i=15 -> 0
        self.assertEqual(mirror_invert(0, 16), 15)
        self.assertEqual(mirror_invert(1, 16), 7)
        self.assertEqual(mirror_invert(2, 16), 11)
        self.assertEqual(mirror_invert(15, 16), 0)
        self.assertEqual(first_bit_set(0, 16), 16)
        self.assertEqual(first_bit_set(1, 16), 0)
        self.assertEqual(first_bit_set(2, 16), 1)
        self.assertEqual(first_bit_set(8, 16), 3)

    def test_pat_aggregation_factor_scaling(self) -> None:
        # Small per-channel payload doubles aggFactor up to nranks/2
        small_params = compute_pat_aggregation_params(
            step_size_bytes=524288,
            step_depth=8,
            max_parallel_factor=8,
            channel_bytes=1024,
            nranks=512,
        )
        self.assertEqual(small_params["agg_factor"], 256)
        self.assertEqual(small_params["agg_delta"], 2)
        self.assertEqual(small_params["binomial_tree_depth"], 9)

    def test_pat_vs_ring_latency_and_bandwidth(self) -> None:
        res = run_pat_experiment()
        summary = res["summary"]
        self.assertGreater(summary["n1024_4mib_speedup_x"], 15.0)
        self.assertGreater(summary["n1024_16mib_speedup_x"], 5.0)
        self.assertEqual(summary["n1024_ring_hops"], 1023)
        self.assertEqual(summary["n1024_pat_inter_node_hops"], 7)

    def test_cli_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, "experiments/nccl_pat_binomial_allgather_rs/run.py", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertTrue(RESULTS_PATH.exists())


if __name__ == "__main__":
    unittest.main()
