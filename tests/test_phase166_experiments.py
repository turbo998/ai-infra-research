import subprocess
import sys
import unittest


class TestPhase166Experiments(unittest.TestCase):
    def test_flashinfer_megamoe_sm103a_fa_sm100_hd256_flashmla_1m_phase166(self):
        subprocess.run(
            [sys.executable, "experiments/flashinfer_megamoe_sm103a_fa_sm100_hd256_flashmla_1m_phase166/run.py", "--check"],
            check=True,
        )

    def test_sglang_mori_aiter_megamoe_sym_vllm_logprobs_deepep_rdc_phase166(self):
        subprocess.run(
            [sys.executable, "experiments/sglang_mori_aiter_megamoe_sym_vllm_logprobs_deepep_rdc_phase166/run.py", "--check"],
            check=True,
        )

    def test_nccl_niin_multictx_rccl_dda_prefetch_nixl_prom_mooncake_phase166(self):
        subprocess.run(
            [sys.executable, "experiments/nccl_niin_multictx_rccl_dda_prefetch_nixl_prom_mooncake_phase166/run.py", "--check"],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
