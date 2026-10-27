"""Unit tests for Phase 113 (priority: 122) deterministic experiment suites."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase113Experiments(unittest.TestCase):
    def test_track_a_flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "flashinfer_frost_sm80_tcgen05_gdp_kda_cutlass_tilelang_phase113"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("OK:", res.stdout)

    def test_track_b_sglang_lora_caller_buf_vllm_kvcr_recency_nixl_trtllm(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "sglang_lora_caller_buf_vllm_kvcr_recency_nixl_trtllm_phase113"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("OK:", res.stdout)

    def test_track_c_te_compile_linear_bias_sm12x_fa4_megatron_rope32_mooncake(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "te_compile_linear_bias_sm12x_fa4_megatron_rope32_mooncake_phase113"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn("OK:", res.stdout)


if __name__ == "__main__":
    unittest.main()
