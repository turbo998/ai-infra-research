import subprocess
import sys
import unittest

class TestPhase163Experiments(unittest.TestCase):
    def test_exp1_flashinfer_dw_fp32_wcx_fa_packgqa_flashmla_abi(self):
        res = subprocess.run([sys.executable, "experiments/flashinfer_dw_fp32_wcx_fa_packgqa_flashmla_abi_phase163/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_exp2_sglang_late_staging_ulysses_3wave_vllm_aiter0127_deepep_v2(self):
        res = subprocess.run([sys.executable, "experiments/sglang_late_staging_ulysses_3wave_vllm_aiter0127_deepep_v2_phase163/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_exp3_nccl_sm100_902_rccl_ll128_regs_nixl_torch216_lmcache_dsv4_hma(self):
        res = subprocess.run([sys.executable, "experiments/nccl_sm100_902_rccl_ll128_regs_nixl_torch216_lmcache_dsv4_hma_phase163/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
