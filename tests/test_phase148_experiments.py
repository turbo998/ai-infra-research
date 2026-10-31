import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase148Experiments(unittest.TestCase):
    def test_track1_flashinfer_triton_tilelang_phase148(self) -> None:
        script = ROOT / "experiments" / "flashinfer_lmhead_rmsnorm_gdn_fmha_triton_tilelang_phase148" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track2_sglang_vllm_trtllm_phase148(self) -> None:
        script = ROOT / "experiments" / "sglang_dsv4_woa_vllm_anthropic_rust_trtllm_responses_phase148" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track3_megatron_pytorch_mooncake_nixl_phase148(self) -> None:
        script = ROOT / "experiments" / "megatron_rl_trainstate_mooncake_part_lmcache_nixl_gds_phase148" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
