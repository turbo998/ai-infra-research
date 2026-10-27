import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase103Experiments(unittest.TestCase):
    def _run_suite(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
        subprocess.run([sys.executable, str(script), "--check"], check=True, cwd=ROOT)

    def test_flashinfer_blackwell_bf16_fp4_w4a16_megamoe_phase103(self) -> None:
        self._run_suite("experiments/flashinfer_blackwell_bf16_fp4_w4a16_megamoe_phase103")

    def test_sglang_dspark_hicache_vllm_trtllm_phase103(self) -> None:
        self._run_suite("experiments/sglang_dspark_hicache_vllm_trtllm_phase103")

    def test_te_row_mxfp8_megatron_muon_mooncake_phase103(self) -> None:
        self._run_suite("experiments/te_row_mxfp8_megatron_muon_mooncake_phase103")


if __name__ == "__main__":
    unittest.main()
