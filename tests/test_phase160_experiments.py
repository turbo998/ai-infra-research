from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase160Experiments(unittest.TestCase):
    def _run_exp(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=str(ROOT))
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=str(ROOT))

    def test_track1_flashinfer_fa4_cutlass_triton_phase160(self) -> None:
        self._run_exp("experiments/flashinfer_gdn_bwd_rubin_megamoe_fa4_swapab_triton_gsan_phase160")

    def test_track2_sglang_vllm_trtllm_deepep_phase160(self) -> None:
        self._run_exp("experiments/sglang_ttft_fill_vllm_mori_ep_trtllm_nvfp4_deepep_gin_phase160")

    def test_track3_nccl_rccl_nixl_mooncake_lmcache_phase160(self) -> None:
        self._run_exp("experiments/nccl_gin_mcst_rccl_bf16_nixl_obj_mooncake_tent_lmcache_phase160")


if __name__ == "__main__":
    unittest.main()
