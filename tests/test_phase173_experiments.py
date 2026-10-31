import subprocess, sys, unittest

class TestPhase173Experiments(unittest.TestCase):
    def test_flashinfer_cake_ticket_zero_cutlass_fp8_gelu_fa4_graph_phase173(self):
        res = subprocess.run([sys.executable, "experiments/flashinfer_cake_ticket_zero_cutlass_fp8_gelu_fa4_graph_phase173/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_vllm_rocm_dsv3_nixl_sglang_dsv4_w8a8_glm52_phase173(self):
        res = subprocess.run([sys.executable, "experiments/vllm_rocm_dsv3_nixl_sglang_dsv4_w8a8_glm52_phase173/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_megatron_mfsdp_shortcut_moe_te_bf16_grouped_mlp_phase173(self):
        res = subprocess.run([sys.executable, "experiments/megatron_mfsdp_shortcut_moe_te_bf16_grouped_mlp_phase173/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
