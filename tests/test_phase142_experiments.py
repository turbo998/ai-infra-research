import subprocess
import sys
import unittest


class TestPhase142Experiments(unittest.TestCase):
    def test_exp1_kernels(self):
        res = subprocess.run(
            [sys.executable, "experiments/flashinfer_megamoe_kimik3_vcattn_triton_tilelang_phase142/run.py", "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"status": "ok"', res.stdout)

    def test_exp2_serving(self):
        res = subprocess.run(
            [sys.executable, "experiments/sglang_w4a16_vllm_nvfp4_pdl_trtllm_serve8_phase142/run.py", "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"status": "ok"', res.stdout)

    def test_exp3_training_fabric(self):
        res = subprocess.run(
            [sys.executable, "experiments/megatron_moonep_mfsdpv2_mooncake_nixl_c2c_phase142/run.py", "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"status": "ok"', res.stdout)


if __name__ == "__main__":
    unittest.main()
