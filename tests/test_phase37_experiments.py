from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase37Experiments(unittest.TestCase):
    def _run_and_check(self, exp_name: str) -> None:
        script = ROOT / "experiments" / exp_name / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_flashinfer_sm12x_w4a16_vllm_kimi_rocm_sglang_unified_kv(self) -> None:
        self._run_and_check("flashinfer_sm12x_w4a16_vllm_kimi_rocm_sglang_unified_kv")

    def test_torchtitan_muse_default_regions_rope_megatron_detok(self) -> None:
        self._run_and_check("torchtitan_muse_default_regions_rope_megatron_detok")

    def test_mooncake_namespace_eviction_lmcache_rbln_dynamo_nixl(self) -> None:
        self._run_and_check("mooncake_namespace_eviction_lmcache_rbln_dynamo_nixl")


if __name__ == "__main__":
    unittest.main()
