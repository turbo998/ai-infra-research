import unittest
from experiments.flashinfer_mxfp_ue8m0_rubin_2cta_moe_qwen38_qsa_phase107.run import run_experiment as run_exp1
from experiments.vllm_rocm_mla_fp8_head_pad_kimi_k3_pd_trtllm_keepalive_phase107.run import run_experiment as run_exp2
from experiments.te_exit_dtor_megatron_fp8_gc_mooncake_egm_nixl_ucx_trace_phase107.run import run_experiment as run_exp3


class TestPhase107Experiments(unittest.TestCase):
    def test_flashinfer_mxfp_rubin_moe_qwen38_qsa(self):
        res = run_exp1()
        self.assertEqual(res["mxfp_ue8m0_subnormal"]["subnormal_e8m0_byte"], 0)
        self.assertEqual(res["mxfp_ue8m0_subnormal"]["subnormal_inverse_scale"], 0.0)
        self.assertTrue(res["rubin_moe_tactics"][2]["valid_single_routing_tile"])
        self.assertEqual(res["rubin_moe_tactics"][2]["a_sfa_multicast_factor"], 2.0)
        self.assertFalse(res["rubin_moe_tactics"][3]["valid_single_routing_tile"])
        self.assertEqual(res["qwen38_qsa_decode"]["bandwidth_speedup_ratio"], 8.0)
        self.assertEqual(res["tilelang_barrier_uniqueness"]["fixed_collision_count"], 0)

    def test_vllm_rocm_mla_fp8_kimi_k3_pd_trtllm_keepalive(self):
        res = run_exp2()
        self.assertFalse(res["rocm_mla_head_padding"][0]["fp8_prefill_enabled"])
        self.assertGreater(res["rocm_mla_head_padding"][0]["speedup_vs_forced_fp8"], 1.35)
        self.assertTrue(res["rocm_mla_head_padding"][1]["fp8_prefill_enabled"])
        self.assertGreater(res["kimi_k3_pd_offload"]["capacity_gain_ratio"], 1.33)
        self.assertEqual(res["trtllm_keepalive_and_mpi_stop"]["emitted_in_progress_keepalives"], 6)
        self.assertFalse(res["trtllm_keepalive_and_mpi_stop"]["dropped_with_keepalive"])

    def test_te_exit_dtor_megatron_gc_mooncake_egm_nixl_trace(self):
        res = run_exp3()
        self.assertTrue(res["te_exit_dtor"]["exit_use_after_free_eliminated"])
        self.assertEqual(res["megatron_fp8_checkpoint_gc"]["saved_wall_ms_per_checkpoint"], 1850.0)
        self.assertTrue(res["mooncake_egm_local_buffer"]["supports_provider_put_and_get_buffer"])
        self.assertEqual(res["mooncake_egm_local_buffer"]["aligned_egm_pool_gb"], 128.0)
        self.assertEqual(res["nixl_ucx_trace"]["single_trace_id_continuity_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
