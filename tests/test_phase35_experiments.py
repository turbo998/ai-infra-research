from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase35Experiments(unittest.TestCase):
    def _run_check(self, rel_script: str) -> None:
        script = ROOT / rel_script
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"{rel_script} failed:\n{proc.stderr}\n{proc.stdout}")
        self.assertIn("OK:", proc.stdout)

    def test_flashinfer_gdn_sm90_sglang_dsv41_trtllm_moe(self) -> None:
        self._run_check("experiments/flashinfer_gdn_sm90_sglang_dsv41_trtllm_moe/run.py")

    def test_torchtitan_muse_spark_compile_megatron_logprobs(self) -> None:
        self._run_check("experiments/torchtitan_muse_spark_compile_megatron_logprobs/run.py")

    def test_mooncake_page_gather_lmcache_scratch_dynamo_kvcr(self) -> None:
        self._run_check("experiments/mooncake_page_gather_lmcache_scratch_dynamo_kvcr/run.py")


if __name__ == "__main__":
    unittest.main()
