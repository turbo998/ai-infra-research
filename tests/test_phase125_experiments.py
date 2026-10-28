from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase125Experiments(unittest.TestCase):
    def _run_suite(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track_a_flashinfer_tilelang_phase125(self) -> None:
        self._run_suite(
            "experiments/flashinfer_deepgemm_mqa_kimi_latent_moe_ssd_sm107_mm_fp4_mc8_sm120_fp8_mhc_dy_pdl_tilelang_phase125"
        )

    def test_track_b_sglang_vllm_trtllm_phase125(self) -> None:
        self._run_suite(
            "experiments/sglang_comfyui_kwargs_status_mm_obs_prefix_slots_vllm_shared_exp_muse_r3_swa_trtllm_phase125"
        )

    def test_track_c_megatron_torchtitan_verl_mooncake_lmcache_dynamo_phase125(self) -> None:
        self._run_suite(
            "experiments/megatron_tp_shared_exp_torchtitan_compile_loss_verl_tpu_grpo_gate_rocm_mooncake_lmcache_dynamo_phase125"
        )


if __name__ == "__main__":
    unittest.main()
