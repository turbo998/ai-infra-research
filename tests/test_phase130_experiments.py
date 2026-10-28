import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase130Experiments(unittest.TestCase):
    def test_track1_flashinfer_mhc_allreduce_fmha_ring_gqa_pt_triton_fp8(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "flashinfer_mhc_allreduce_fmha_ring_gqa_pt_triton_fp8_phase130"
            / "run.py"
        )
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"glm53_mhc_saves_89_launches_per_decode_step": true', res.stdout)
        self.assertIn('"cute_dsl_5stage_pt_ring_eliminates_race": true', res.stdout)

    def test_track2_sglang_engram_decode_dcp_mla_hicache_vllm_dbo_trtllm(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "sglang_engram_decode_dcp_mla_hicache_vllm_dbo_trtllm_phase130"
            / "run.py"
        )
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"engram_all_to_all_reduces_decode_comm_by_16x": true', res.stdout)
        self.assertIn('"vllm_dbo_threshold_7_clamped_to_8_for_3_drafts": true', res.stdout)

    def test_track3_megatron_dp1_rs_pytorch_rocsolver_mooncake_lmcache_dynamo_epp(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "megatron_dp1_rs_pytorch_rocsolver_mooncake_lmcache_dynamo_epp_phase130"
            / "run.py"
        )
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"megatron_dp1_eliminates_512mib_fp32_buffer": true', res.stdout)
        self.assertIn('"lmcache_repeated_close_eliminates_11s_hang": true', res.stdout)


if __name__ == "__main__":
    unittest.main()
