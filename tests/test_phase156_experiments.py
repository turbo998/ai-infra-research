import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase156Experiments(unittest.TestCase):
    def test_track1_gpu_compiler_phase156(self):
        script = ROOT / "experiments" / "flashinfer_megamoe_sm103a_mhc_triton_barriers_tilelang_fp4_phase156" / "run.py"
        p = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_track2_serving_moe_phase156(self):
        script = ROOT / "experiments" / "sglang_pp_kv_consensus_vllm_hisparse_trtllm_qwen38_spec_phase156" / "run.py"
        p = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_track3_training_kv_fabric_phase156(self):
        script = ROOT / "experiments" / "megatron_fsdp_layout_rl_logprobs_mooncake_efa_nixl_ucx_phase156" / "run.py"
        p = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == "__main__":
    unittest.main()
