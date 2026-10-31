import unittest
from experiments.flashinfer_sm107a_mhc_causal_conv1d_fa_sm100_mask_phase171.run import run_model as run_fi
from experiments.vllm_template_trust_kimi_vit_sglang_xpu_sched_phase171.run import run_model as run_vllm
from experiments.nccl_ompi_strip_rocjitsu_fp64_nixl_logfile_phase171.run import run_model as run_nccl


class TestPhase171Experiments(unittest.TestCase):
    def test_flashinfer_sm107a_mhc_causal_conv1d_fa_sm100_mask(self):
        res = run_fi()
        self.assertEqual(res["sm107a_min_cuda_version"], "13.4")
        self.assertEqual(res["mask_compile_speedup_x"], 5.0)
        self.assertEqual(res["sm80_sm120_invalid_varlen_tile_oob"], 0)
        self.assertEqual(res["reused_coalesce_analysis_passes"], 1)

    def test_vllm_template_trust_kimi_vit_sglang_xpu_sched(self):
        res = run_vllm()
        self.assertEqual(res["untrusted_template_injections_accepted"], 0)
        self.assertTrue(res["non_jinja_inline_system_probed"])
        self.assertEqual(res["kimi_k25_vit_eager_compile_stalls"], 0)
        self.assertEqual(res["dead_decoded_text_fields_removed"], 2)

    def test_nccl_ompi_strip_rocjitsu_fp64_nixl_logfile(self):
        res = run_nccl()
        self.assertEqual(res["inherited_ompi_env_vars_in_child"], 0)
        self.assertEqual(res["cross_test_state_leaks"], 0)
        self.assertEqual(res["max_rotated_log_files_per_process"], 2)


if __name__ == "__main__":
    unittest.main()
