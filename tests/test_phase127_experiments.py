from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase127Experiments(unittest.TestCase):
    def _run_suite(self, rel_dir: str) -> None:
        script = ROOT / "experiments" / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track_a_flashinfer_cake_lamport_b12x_phase127(self) -> None:
        self._run_suite("flashinfer_cake_softmax_kimi_k3_lamport_b12x_e8m0_tilelang_triton_phase127")

    def test_track_b_sglang_vllm_trtllm_phase127(self) -> None:
        self._run_suite("sglang_unified_kv_subpool_idspace_pricing_vllm_nvfp4_trtllm_minimax_phase127")

    def test_track_c_megatron_pytorch_mooncake_dynamo_phase127(self) -> None:
        self._run_suite("megatron_sqrelu_pytorch_online_softmax_pdl_mooncake_dynamo_gds_phase127")


if __name__ == "__main__":
    unittest.main()
