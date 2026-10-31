import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase147Experiments(unittest.TestCase):
    def test_track1_flashinfer_triton_tilelang_phase147(self) -> None:
        script = ROOT / "experiments" / "flashinfer_cake_qsa_gdn_msa_triton_tilelang_phase147" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track2_sglang_vllm_trtllm_phase147(self) -> None:
        script = ROOT / "experiments" / "sglang_amd_aiter_vllm_qwen4_rust_trtllm_ltx23_phase147" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track3_megatron_pytorch_mooncake_nixl_phase147(self) -> None:
        script = ROOT / "experiments" / "megatron_cg_cp_nofork_mooncake_s3_lmcache_nixl_phase147" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
