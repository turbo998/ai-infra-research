import subprocess
import sys
import unittest

class TestPhase177Experiments(unittest.TestCase):
    def test_flashinfer_sm107a_ue8m0_gdn_sm120_fa_sink_bwd_phase177(self):
        res = subprocess.run([sys.executable, "experiments/flashinfer_sm107a_ue8m0_gdn_sm120_fa_sink_bwd_phase177/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_vllm_rocm_skinny_gemm_kimi_k3_kda_sglang_muse_spark_phase177(self):
        res = subprocess.run([sys.executable, "experiments/vllm_rocm_skinny_gemm_kimi_k3_kda_sglang_muse_spark_phase177/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_megatron_gdp_fp32_state_te_grouped_cpu_nccl_encryption_phase177(self):
        res = subprocess.run([sys.executable, "experiments/megatron_gdp_fp32_state_te_grouped_cpu_nccl_encryption_phase177/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
