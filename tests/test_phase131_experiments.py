import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase131Experiments(unittest.TestCase):
    def _run_check(self, rel_dir: str) -> None:
        script = ROOT / rel_dir / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, msg=res.stderr + res.stdout)

    def test_track_a_phase131(self) -> None:
        self._run_check("experiments/flashinfer_cake_fp4_conv_gdn_triton_warp_fold_cdna_phase131")

    def test_track_b_phase131(self) -> None:
        self._run_check("experiments/sglang_engram_nvlink_mori_gfx950_vllm_helion_trtllm_video_phase131")

    def test_track_c_phase131(self) -> None:
        self._run_check("experiments/megatron_streamwise_16x512_pytorch_rocprof_mooncake_lmcache_dynamo_phase131")


if __name__ == "__main__":
    unittest.main()
