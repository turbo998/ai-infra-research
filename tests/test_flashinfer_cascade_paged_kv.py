"""Unit tests for FlashInfer Multi-Level Cascade Attention & Base-2 LSE State Merging Simulator."""

from __future__ import annotations

import subprocess
import sys
import unittest

from experiments.flashinfer_cascade_paged_kv.run import (
    RESULTS_PATH,
    compute_segment_attention_base2,
    merge_state_in_place_base2,
    simulate_cascade_paged_kv_sweep,
    verify_multi_level_cascade_numerics,
)


class TestFlashInferCascadePagedKV(unittest.TestCase):
    def test_base2_lse_merge_numerical_equivalence(self) -> None:
        numerics = verify_multi_level_cascade_numerics()
        self.assertLess(numerics["max_output_abs_error"], 1e-14)
        self.assertLess(numerics["lse_abs_error"], 1e-14)
        self.assertEqual(numerics["empty_segment_guard_error"], 0.0)
        self.assertTrue(numerics["both_empty_returns_neg_inf"])

    def test_empty_prefix_neg_inf_guard(self) -> None:
        q = [0.5, -0.25, 0.75, 0.1]
        v_empty, s_empty = compute_segment_attention_base2(q, [], [], 0.5)
        self.assertTrue(s_empty == float("-inf"))
        v_valid, s_valid = compute_segment_attention_base2(
            q, [[0.2, 0.1, -0.3, 0.4]], [[1.0, 2.0, 3.0, 4.0]], 0.5
        )
        v_merged, s_merged = merge_state_in_place_base2(v_empty, s_empty, v_valid, s_valid)
        for a, b in zip(v_merged, v_valid):
            self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(s_merged, s_valid, places=12)

    def test_cascade_batch_sweep_monotonic_reduction(self) -> None:
        res = simulate_cascade_paged_kv_sweep()
        summary = res["summary"]
        self.assertGreater(summary["b128_hbm_reduction_pct"], 94.0)
        self.assertGreater(summary["b128_speedup_x"], 15.0)
        self.assertGreater(
            summary["b128_ai_cascade3_flops_per_byte"],
            summary["b128_ai_flat_flops_per_byte"] * 15.0,
        )

    def test_cli_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, "experiments/flashinfer_cascade_paged_kv/run.py", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertTrue(RESULTS_PATH.exists())


if __name__ == "__main__":
    unittest.main()
