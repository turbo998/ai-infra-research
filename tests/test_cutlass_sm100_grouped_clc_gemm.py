"""Unit tests for Phase 17 experiment: `cutlass_sm100_grouped_clc_gemm`."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest

from experiments.cutlass_sm100_grouped_clc_gemm.run import (
    generate_zipf_expert_tokens,
    simulate_grouped_gemm_schedule,
    validate_summary,
)


class TestCutlassSm100GroupedClcGemm(unittest.TestCase):
    def test_zipf_token_conservation(self) -> None:
        counts = generate_zipf_expert_tokens(64, 8192, 1.15)
        self.assertEqual(len(counts), 64)
        self.assertEqual(sum(counts), 8192)
        self.assertGreater(counts[0], counts[-1] * 20)

    def test_schedule_monotonic_hierarchy_and_tma_stall_reduction(self) -> None:
        summary = simulate_grouped_gemm_schedule()
        validate_summary(summary)
        p64 = summary["policy_comparison_64e"]
        self.assertLess(
            p64["async_tma_clc_dynamic"]["wall_time_us"],
            p64["async_tma_static_group"]["wall_time_us"],
        )
        self.assertLess(
            p64["async_tma_static_group"]["wall_time_us"],
            p64["sync_tma_static_group"]["wall_time_us"],
        )
        self.assertLess(
            p64["async_tma_clc_dynamic"]["tma_update_stall_us"],
            1.0,
        )
        self.assertGreater(
            p64["sync_tma_static_group"]["tma_update_stall_us"],
            10.0,
        )

    def test_shared_storage_tensormap_alignment_and_cli_check(self) -> None:
        summary = simulate_grouped_gemm_schedule()
        layout = summary["shared_storage_layout"]
        self.assertEqual(layout["tensormap_bytes_per_descriptor"], 128)
        self.assertEqual(layout["total_tensormap_smem_bytes"], 1024)
        self.assertEqual(layout["clc_response_bytes_async"], 128)

        script = (
            pathlib.Path(__file__).resolve().parents[1]
            / "experiments"
            / "cutlass_sm100_grouped_clc_gemm"
            / "run.py"
        )
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS] cutlass_sm100_grouped_clc_gemm", proc.stdout)
        summary_file = script.parent / "results" / "summary.json"
        self.assertTrue(summary_file.exists())
        loaded = json.loads(summary_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["Check"], "PASS")


if __name__ == "__main__":
    unittest.main()
