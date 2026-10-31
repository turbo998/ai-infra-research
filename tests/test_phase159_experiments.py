from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase159Experiments(unittest.TestCase):
    def _run_check(self, rel_dir: str) -> None:
        script = ROOT / "experiments" / rel_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)

    def test_flashinfer_kimi_k3_fmha_fp8_sm120_mqa_triton_fp4_phase159(self) -> None:
        self._run_check("flashinfer_kimi_k3_fmha_fp8_sm120_mqa_triton_fp4_phase159")

    def test_sglang_diffusion_subtile_vllm_weight_clone_trtllm_gdn_phase159(self) -> None:
        self._run_check("sglang_diffusion_subtile_vllm_weight_clone_trtllm_gdn_phase159")

    def test_nccl_gin_coop_rccl_dmabuf_nixl_ep_mooncake_lmcache_phase159(self) -> None:
        self._run_check("nccl_gin_coop_rccl_dmabuf_nixl_ep_mooncake_lmcache_phase159")


if __name__ == "__main__":
    unittest.main()
