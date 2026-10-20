from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase36Experiments(unittest.TestCase):
    def _run_check(self, rel_dir: str) -> None:
        script = ROOT / "experiments" / rel_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=f"{rel_dir} failed:\n{res.stderr}\n{res.stdout}")

    def test_flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile(self) -> None:
        self._run_check("flashinfer_causal_conv1d_bwd_trtllm_sm107_vllm_compile")

    def test_torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun(self) -> None:
        self._run_check("torchtitan_fsdp_inspect_kimi_k3_mxfp4_megatron_rerun")

    def test_sglang_uncached_bucket_mooncake_inflight_lmcache_apc(self) -> None:
        self._run_check("sglang_uncached_bucket_mooncake_inflight_lmcache_apc")


if __name__ == "__main__":
    unittest.main()
