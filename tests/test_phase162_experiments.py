from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase162Experiments(unittest.TestCase):
    def _run_exp(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track1_flashinfer_det_loss_rmsnorm_fa_triton(self) -> None:
        self._run_exp("experiments/flashinfer_det_loss_rmsnorm_fa_aiter_triton_mir_phase162")

    def test_track2_sglang_m3_ltx25_vllm_mi355_trtllm_deepep(self) -> None:
        self._run_exp("experiments/sglang_m3_enum_ltx25_vllm_gb300_mi355_trtllm_deepep_sm90_phase162")

    def test_track3_nccl4py_rccl_nixl_mooncake_lmcache_xpu(self) -> None:
        self._run_exp("experiments/nccl4py_2323_rccl_nixl_mooncake_tent_tone_lmcache_xpu_phase162")


if __name__ == "__main__":
    unittest.main()
