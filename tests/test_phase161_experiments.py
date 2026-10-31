from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase161Experiments(unittest.TestCase):
    def _run_exp(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track1_flashinfer_cake_mqa_mhc_fa_cutlass_triton(self) -> None:
        self._run_exp("experiments/flashinfer_cake_mqa_mhc_fa_sm100_cutlass_48_triton_proton_phase161")

    def test_track2_sglang_joint_lru_dp_abort_vllm_trtllm(self) -> None:
        self._run_exp("experiments/sglang_joint_lru_dp_abort_vllm_nixl152_trtllm_own_frag_phase161")

    def test_track3_nccl_rccl_dmabuf_amdsmi_nixl_mooncake_lmcache(self) -> None:
        self._run_exp("experiments/nccl_2323_rccl_dmabuf_amdsmi_nixl_gpunetio_mooncake_lmcache_phase161")


if __name__ == "__main__":
    unittest.main()
