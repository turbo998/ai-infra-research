import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase80Experiments(unittest.TestCase):
    def _run_exp(self, rel_dir: str) -> None:
        script = ROOT / "experiments" / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track_a_flashinfer_megamoe_ep16_nvfp4_mla_sage_sm120(self) -> None:
        self._run_exp("flashinfer_mxfp8_megamoe_ep16_relaxed_route_nvfp4_mla_paced_dequant_sage_sm120")

    def test_track_b_sglang_minimax_m3_dflash_dp_vllm_modelopt_mixed(self) -> None:
        self._run_exp("sglang_minimax_m3_sparse_fp8_dflash_dp_attn_draft_layout_vllm_modelopt_mixed")

    def test_track_c_fa4_sp_pingpong_cutlass_clc_verl_qat_slime(self) -> None:
        self._run_exp("fa4_sm100_sp_pingpong_hd256_2cta_bwd_cutlass_clc_epi_verl_qat_slime")


if __name__ == "__main__":
    unittest.main()
