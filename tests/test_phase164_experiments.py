import subprocess
import sys
import unittest

class TestPhase164Experiments(unittest.TestCase):
    def test_exp1_flashinfer_route_pref_fa_bwd_zero_flashmla_v41(self):
        res = subprocess.run([sys.executable, "experiments/flashinfer_route_pref_fa_bwd_zero_flashmla_v41_phase164/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_exp2_sglang_pd_sampler_blake3_vllm_mxfp4_deepep_internode(self):
        res = subprocess.run([sys.executable, "experiments/sglang_pd_sampler_blake3_vllm_mxfp4_deepep_internode_phase164/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_exp3_nccl_efa_gda_niin_rocshmem_nixl_cuobject_lmcache_l2_shm(self):
        res = subprocess.run([sys.executable, "experiments/nccl_efa_gda_niin_rocshmem_nixl_cuobject_lmcache_l2_shm_phase164/run.py", "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
