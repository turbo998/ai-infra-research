"""Unit tests for experiments/nccl_zero_copy_registration."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.nccl_zero_copy_registration.run import RESULTS_PATH, build_results


class TestNcclZeroCopyRegistration(unittest.TestCase):
    def test_deterministic_results_match(self) -> None:
        payload = build_results()
        on_disk = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload, on_disk)
        self.assertEqual(payload["measurement_class"], "own-offline")

    def test_sm_preemption_and_wave_quantization(self) -> None:
        payload = build_results()
        self.assertEqual(payload["summary"]["zero_copy_2cta_gemm_slowdown_pct"], 0.0)
        self.assertGreater(payload["summary"]["unreg_32cta_gemm_slowdown_pct"], 30.0)
        rows = {r["mode"]: r for r in payload["sm_preemption_wave_quantization"]}
        self.assertEqual(rows["zero_copy_nvls_window_reg"]["gemm_waves_for_1024_ctas"], 8)
        self.assertGreaterEqual(rows["unreg_tree_nvls_24_channels"]["gemm_waves_for_1024_ctas"], 10)

    def test_buffer_pool_and_p99_latency_bounds(self) -> None:
        payload = build_results()
        self.assertGreaterEqual(payload["summary"]["min_hbm_saved_mib"], 576.0)
        self.assertGreaterEqual(payload["summary"]["max_hbm_saved_gib"], 2.3)
        self.assertGreaterEqual(payload["summary"]["max_p99_speedup_x"], 3.5)

    def test_cli_check_passes(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "nccl_zero_copy_registration" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
