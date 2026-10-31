import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase157Experiments(unittest.TestCase):
    def test_track1_gpu_compiler_phase157(self):
        script = ROOT / "experiments" / "flashinfer_tns_detloss_mhc_sm107a_allreduce_triton_tilelang_phase157" / "run.py"
        p = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_track2_serving_moe_phase157(self):
        script = ROOT / "experiments" / "sglang_page_envelope_qwen3_dsa_vllm_skinny_reload_trtllm_phase157" / "run.py"
        p = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_track3_training_kv_fabric_phase157(self):
        script = ROOT / "experiments" / "megatron_mxfp8_refit_sigmoid_rocm_mooncake_fmix64_nixl_phase157" / "run.py"
        p = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()
