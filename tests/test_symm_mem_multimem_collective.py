"""Unit tests for Phase 17 experiment: `symm_mem_multimem_collective`."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest

from experiments.symm_mem_multimem_collective.run import (
    model_algorithm_latency_us,
    run_simulation,
    simulate_numerical_accumulation_determinism,
    validate_summary,
)


class TestSymmMemMultimemCollective(unittest.TestCase):
    def test_traffic_multipliers_and_sm_footprints(self) -> None:
        w = 8
        msg = 16 * 1024 * 1024
        one_shot = model_algorithm_latency_us(msg, w, "symm_one_shot_p2p")
        two_shot = model_algorithm_latency_us(msg, w, "symm_two_shot_p2p")
        mm_2shot = model_algorithm_latency_us(msg, w, "symm_multimem_two_shot_inplace")

        self.assertAlmostEqual(one_shot["nvlink_traffic_multiplier"], 7.0, places=4)
        self.assertAlmostEqual(two_shot["nvlink_traffic_multiplier"], 1.75, places=4)
        self.assertAlmostEqual(mm_2shot["nvlink_traffic_multiplier"], 1.125, places=4)
        self.assertEqual(mm_2shot["sm_blocks_used"], 8)
        self.assertEqual(two_shot["sm_blocks_used"], 24)

    def test_numerical_acc_f32_vs_bf16_and_determinism(self) -> None:
        num = simulate_numerical_accumulation_determinism(world_size=8, num_trials=128)
        self.assertLess(
            num["multimem_acc_f32_mean_abs_err"],
            num["p2p_load_and_reduce_bf16_mean_abs_err"],
        )
        self.assertGreater(num["error_reduction_ratio_acc_f32_vs_bf16_add"], 2.0)
        self.assertTrue(num["p2p_load_and_reduce_cross_rank_bitwise_identical"])
        self.assertTrue(num["multimem_two_shot_st_cross_rank_bitwise_identical"])

    def test_cli_check_and_summary_output(self) -> None:
        summary = run_simulation()
        validate_summary(summary)
        script = (
            pathlib.Path(__file__).resolve().parents[1]
            / "experiments"
            / "symm_mem_multimem_collective"
            / "run.py"
        )
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] symm_mem_multimem_collective", proc.stdout)
        summary_file = script.parent / "results" / "summary.json"
        loaded = json.loads(summary_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["Check"], "PASS")


if __name__ == "__main__":
    unittest.main()
