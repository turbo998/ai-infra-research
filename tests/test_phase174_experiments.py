import subprocess, sys, unittest

class TestPhase174Experiments(unittest.TestCase):
    def test_flashinfer_cake_sm100_dsv4_moe_twoshot_triton_volatile_phase174(self):
        res = subprocess.run([sys.executable, "experiments/flashinfer_cake_sm100_dsv4_moe_twoshot_triton_volatile_phase174/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_vllm_mrv2_spec_decode_hyv4_sglang_minimax_m31_phase174(self):
        res = subprocess.run([sys.executable, "experiments/vllm_mrv2_spec_decode_hyv4_sglang_minimax_m31_phase174/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_megatron_gdp_gtp_inplace_te_cudnn_grouped_nccl_ras_phase174(self):
        res = subprocess.run([sys.executable, "experiments/megatron_gdp_gtp_inplace_te_cudnn_grouped_nccl_ras_phase174/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
