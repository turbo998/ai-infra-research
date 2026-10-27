import unittest
from experiments.flashinfer_cake_fmha_qsa_ssd_sm90_megamoe_phase108.run import run_experiment as run_exp1
from experiments.sglang_thp_hicache_vllm_nixl_ssm_trtllm_blob_phase108.run import run_experiment as run_exp2
from experiments.megatron_etp_te_cutedsl_torchtitan_ema_mooncake_tent_nixl_phase108.run import run_experiment as run_exp3


class TestPhase108Experiments(unittest.TestCase):
    def test_flashinfer_cake_fmha_qsa_cudnn_cutlass_tilelang(self):
        res = run_exp1()
        self.assertEqual(res["cake_fmha_paged_context"]["programs_registered"], 20)
        self.assertEqual(res["cake_fmha_paged_context"]["nvfp4_vs_bf16_bandwidth_ratio"], 3.5556)
        self.assertEqual(res["cake_qsa_workspace_reuse"]["reused_checks_per_1000_steps"], 1)
        self.assertFalse(res["cake_qsa_workspace_reuse"]["single_split_allocates_partials"])
        self.assertEqual(res["cudnn_ragged_token_offset"]["leaked_padding_rows_eliminated"], 35)
        self.assertEqual(res["cutlass_scatterd_tilelang_k_pad"]["cutlass_fixed_scatterd_offset_bytes"], 0)
        self.assertTrue(res["cutlass_scatterd_tilelang_k_pad"]["tilelang_rejects_unpadded_partial_k"])

    def test_sglang_thp_hicache_vllm_nixl_ssm_trtllm_blob(self):
        res = run_exp2()
        self.assertEqual(res["sglang_hicache_thp"]["page_fault_reduction_factor"], 512)
        self.assertGreater(res["sglang_hicache_thp"]["prefault_speedup_ratio"], 15.0)
        self.assertEqual(res["sglang_dp_attention_pad"]["fixed_dsa_metadata_rows"], 4)
        self.assertEqual(res["sglang_dp_attention_pad"]["dsa_row_corruption_eliminated"], 4)
        self.assertEqual(res["vllm_nixl_hybrid_ssm"]["filtered_transfer_mib"], 96.0)
        self.assertGreater(res["trtllm_blob_and_moe"]["registration_speedup_ratio"], 8.5)
        self.assertTrue(res["trtllm_blob_and_moe"]["force_moe_deterministic_disables_finalize_fusion"])

    def test_megatron_etp_te_cutedsl_torchtitan_ema_mooncake_nixl(self):
        res = run_exp3()
        self.assertEqual(res["megatron_inference_etp"]["tp_ep_world_size"], 16)
        self.assertEqual(res["megatron_inference_etp"]["per_gpu_expert_memory_reduction_ratio"], 2.0)
        self.assertEqual(res["te_cutedsl_eval_gemm"]["activation_quant_memory_saved_pct"], 50.0)
        self.assertTrue(res["te_cutedsl_eval_gemm"]["cutedsl_dense_gemm_enabled_without_colwise"])
        self.assertTrue(res["torchtitan_ema_resharding"]["dcp_planner_accepts_fixed"])
        self.assertTrue(res["mooncake_restore_and_nixl_trace"]["late_restore_rejected_in_serving"])
        self.assertEqual(res["mooncake_restore_and_nixl_trace"]["nixl_terminal_trace_completeness_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
