import subprocess, sys, unittest

class TestPhase175Experiments(unittest.TestCase):
    def test_flashinfer_grouped_fp8_128x64_sm120_burst_fa_sm100_splitkv_phase175(self):
        res = subprocess.run([sys.executable, "experiments/flashinfer_grouped_fp8_128x64_sm120_burst_fa_sm100_splitkv_phase175/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_sglang_hicache_protected_credit_trtllm_row_norm_lmcache_det_kv_phase175(self):
        res = subprocess.run([sys.executable, "experiments/sglang_hicache_protected_credit_trtllm_row_norm_lmcache_det_kv_phase175/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_te_delayed_wgrad_tied_params_nccl_pat_uneven_ranks_phase175(self):
        res = subprocess.run([sys.executable, "experiments/te_delayed_wgrad_tied_params_nccl_pat_uneven_ranks_phase175/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
