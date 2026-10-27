import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase106Experiments(unittest.TestCase):
    def _run_check(self, rel_script: str) -> None:
        script_path = ROOT / rel_script
        res = subprocess.run(
            [sys.executable, str(script_path), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
        self.assertIn("OK:", res.stdout)

    def test_flashinfer_tinygemm2_tcgen05_minimax_msa_warp_decode_phase106(self) -> None:
        self._run_check(
            "experiments/flashinfer_tinygemm2_tcgen05_minimax_msa_warp_decode_phase106/run.py"
        )

    def test_sglang_qwen3_next_rocm_vllm_kimi_k3_mla_trtllm_spec_phase106(self) -> None:
        self._run_check(
            "experiments/sglang_qwen3_next_rocm_vllm_kimi_k3_mla_trtllm_spec_phase106/run.py"
        )

    def test_te_cp_p2p_o1_megatron_gtp_fp32_ubr_mooncake_reshard_phase106(self) -> None:
        self._run_check(
            "experiments/te_cp_p2p_o1_megatron_gtp_fp32_ubr_mooncake_reshard_phase106/run.py"
        )


if __name__ == "__main__":
    unittest.main()
