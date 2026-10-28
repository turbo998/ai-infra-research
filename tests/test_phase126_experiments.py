from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase126Experiments(unittest.TestCase):
    def _run_suite(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track_a_flashinfer_tilelang_phase126(self) -> None:
        self._run_suite(
            "experiments/flashinfer_grouped_fp8_cudagraph_decode_nvfp4_sf_sm12x_moe_sm90_sf_ring_tilelang_phase126"
        )

    def test_track_b_sglang_vllm_trtllm_phase126(self) -> None:
        self._run_suite(
            "experiments/sglang_kv_lifecycle_byte_storage_pins_vllm_sparse_mla_65535_r3_trtllm_kimi_k3_phase126"
        )

    def test_track_c_megatron_torchtitan_verl_mooncake_lmcache_dynamo_phase126(self) -> None:
        self._run_suite(
            "experiments/megatron_get_batch_seqpack_refit_dp_torchtitan_loss_verl_veomni_mooncake_lmcache_dynamo_phase126"
        )


if __name__ == "__main__":
    unittest.main()
