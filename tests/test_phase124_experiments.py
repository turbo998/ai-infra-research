"""Unit tests for Phase 124 experiments."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRACK_A = ROOT / "experiments" / "flashinfer_kda_plan_ssd_sm107_muse_mm_fp4_silu_bf16fp4_sm120_gdn_softmax_mhc_tilelang_phase124" / "run.py"
TRACK_B = ROOT / "experiments" / "sglang_comfyui_kv_storage_routed_views_vllm_p2p_retain_swa_dsa_nixl_rocm_trtllm_phase124" / "run.py"
TRACK_C = ROOT / "experiments" / "megatron_gdp_fp32_bsize_refit_torchtitan_compile_loss_verl_tpu_rocm_veomni_mooncake_lmcache_dynamo_phase124" / "run.py"


class TestPhase124Experiments(unittest.TestCase):
    def test_track_a_flashinfer_tilelang_phase124(self) -> None:
        subprocess.run([sys.executable, str(TRACK_A)], check=True)
        subprocess.run([sys.executable, str(TRACK_A), "--check"], check=True)

    def test_track_b_sglang_vllm_trtllm_phase124(self) -> None:
        subprocess.run([sys.executable, str(TRACK_B)], check=True)
        subprocess.run([sys.executable, str(TRACK_B), "--check"], check=True)

    def test_track_c_megatron_torchtitan_verl_mooncake_lmcache_dynamo_phase124(self) -> None:
        subprocess.run([sys.executable, str(TRACK_C)], check=True)
        subprocess.run([sys.executable, str(TRACK_C), "--check"], check=True)


if __name__ == "__main__":
    unittest.main()
