import subprocess
import sys
import unittest

class TestPhase178Experiments(unittest.TestCase):
    def test_flashinfer_dsa_global_ticket_silu_rmsnorm_fa_sparse_mla_phase178(self):
        res = subprocess.run([sys.executable, "experiments/flashinfer_dsa_global_ticket_silu_rmsnorm_fa_sparse_mla_phase178/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_vllm_mi355x_dsv41_mistral4_p2p_handshake_sglang_gfx1250_phase178(self):
        res = subprocess.run([sys.executable, "experiments/vllm_mi355x_dsv41_mistral4_p2p_handshake_sglang_gfx1250_phase178/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_te_grouped_split_device_align_nccl_allgather_a2a_topo_phase178(self):
        res = subprocess.run([sys.executable, "experiments/te_grouped_split_device_align_nccl_allgather_a2a_topo_phase178/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
