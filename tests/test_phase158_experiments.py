import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase158Experiments(unittest.TestCase):
    def test_track1_flashinfer_mqa_pdl_tinygemm2_conv1d_phase158(self) -> None:
        script = ROOT / "experiments" / "flashinfer_mqa_pdl_tinygemm2_conv1d_mhc_triton_tilelang_phase158" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr or res.stdout)

    def test_track2_sglang_pd_aux_mxfp8_vllm_nixl_trtllm_phase158(self) -> None:
        script = ROOT / "experiments" / "sglang_pd_aux_mxfp8_ulysses_vllm_nixl_pcp_trtllm_phase158" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr or res.stdout)

    def test_track3_megatron_pytorch_mooncake_lmcache_nixl_phase158(self) -> None:
        script = ROOT / "experiments" / "megatron_pretrain_inductor_avx2_mooncake_tls_lmcache_nixl_phase158" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr or res.stdout)


if __name__ == "__main__":
    unittest.main()
