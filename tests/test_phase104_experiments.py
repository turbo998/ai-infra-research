from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase104Experiments(unittest.TestCase):
    def _run_exp(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track_a_flashinfer_conv1d_attnres_fp8_phase104(self) -> None:
        self._run_exp("experiments/flashinfer_causal_conv1d_kimi_attnres_grouped_fp8_phase104")

    def test_track_b_sglang_vllm_trtllm_phase104(self) -> None:
        self._run_exp("experiments/sglang_custom_ar_vmm_vllm_b128_trtllm_lowm_fp8_phase104")

    def test_track_c_te_megatron_torchtitan_mooncake_phase104(self) -> None:
        self._run_exp("experiments/te_moe_dsv4_routing_megatron_vlm_mooncake_batch_phase104")


if __name__ == "__main__":
    unittest.main()
