import subprocess
import sys
import unittest


class TestPhase143Experiments(unittest.TestCase):
    def test_exp1_kernels(self):
        res = subprocess.run(
            [sys.executable, "experiments/flashinfer_xqa_mtp_mxfp4_rubin_mla_triton_phase143/run.py", "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"status": "ok"', res.stdout)

    def test_exp2_serving(self):
        res = subprocess.run(
            [sys.executable, "experiments/sglang_rust_meta_vllm_rs_dplb_trtllm_audio_phase143/run.py", "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"status": "ok"', res.stdout)

    def test_exp3_training_fabric(self):
        res = subprocess.run(
            [sys.executable, "experiments/megatron_embed_wgrad_mooncake_gds_nixl_ucx_phase143/run.py", "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"status": "ok"', res.stdout)


if __name__ == "__main__":
    unittest.main()
