import subprocess, sys, unittest

class TestPhase176Experiments(unittest.TestCase):
    def test_flashinfer_nvfp4_sm103_sm120_cutlass_tilen_fa4_first_tile_phase176(self):
        res = subprocess.run([sys.executable, "experiments/flashinfer_nvfp4_sm103_sm120_cutlass_tilen_fa4_first_tile_phase176/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_vllm_gb300_kimi_k3_sglang_mi355x_qwen38_dcp4_phase176(self):
        res = subprocess.run([sys.executable, "experiments/vllm_gb300_kimi_k3_sglang_mi355x_qwen38_dcp4_phase176/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_te_det_moe_router_no_atomics_ep_handle_nccl_win_pool_phase176(self):
        res = subprocess.run([sys.executable, "experiments/te_det_moe_router_no_atomics_ep_handle_nccl_win_pool_phase176/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
