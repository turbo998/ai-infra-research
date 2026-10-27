import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase105Experiments(unittest.TestCase):
    def _run_experiment(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)

    def test_track_a_flashinfer_mhc_post_pre_tp12_tail_softmax_svdquant_phase105(self) -> None:
        self._run_experiment(
            "experiments/flashinfer_mhc_post_pre_tp12_tail_softmax_svdquant_phase105"
        )

    def test_track_b_sglang_kvshard_glm_fmha_vllm_rocm_dsa_trtllm_phase105(self) -> None:
        self._run_experiment(
            "experiments/sglang_kvshard_glm_fmha_vllm_rocm_dsa_trtllm_phase105"
        )

    def test_track_c_te_nvrtc_nvfp4_ep_fused_megatron_dbuffer_mooncake_phase105(self) -> None:
        self._run_experiment(
            "experiments/te_nvrtc_nvfp4_ep_fused_megatron_dbuffer_mooncake_phase105"
        )


if __name__ == "__main__":
    unittest.main()
