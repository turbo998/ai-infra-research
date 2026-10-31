import subprocess
import sys
import unittest


class TestPhase165Experiments(unittest.TestCase):
    def test_flashinfer_ssd_seq_idx_fa_bwd_swapab_flashmla_ascend_phase165(self):
        subprocess.run(
            [sys.executable, "experiments/flashinfer_ssd_seq_idx_fa_bwd_swapab_flashmla_ascend_phase165/run.py", "--check"],
            check=True,
        )

    def test_sglang_muse_2shot_nomas_kv_vllm_umbp_mori_deepep_v2_phase165(self):
        subprocess.run(
            [sys.executable, "experiments/sglang_muse_2shot_nomas_kv_vllm_umbp_mori_deepep_v2_phase165/run.py", "--check"],
            check=True,
        )

    def test_nccl_niin_tma_local_rocjitsu_py_nixl_gpunetio_mooncake_phase165(self):
        subprocess.run(
            [sys.executable, "experiments/nccl_niin_tma_local_rocjitsu_py_nixl_gpunetio_mooncake_phase165/run.py", "--check"],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
