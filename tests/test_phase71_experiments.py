import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase71Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        script = REPO_ROOT / "experiments" / rel_dir / "run.py"
        summary_file = REPO_ROOT / "experiments" / rel_dir / "summary.json"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        return json.loads(summary_file.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_rubin_mhc_wsmem_sm120_fa2_mtp_vc_tilelang(self) -> None:
        data = self._run_and_load("flashinfer-rubin-mhc-wsmem-sm120-fp8-fa2-mtp-vc-tilelang")
        self.assertEqual(data["phase"], 71)
        self.assertEqual(data["priority"], 80)

        mhc = data["mega_mhc_h4096_wsmem"]
        self.assertEqual(mhc["t256_case"]["selected_regime"], "bf16")
        self.assertEqual(mhc["t2048_case"]["selected_regime"], "tailpipe")
        self.assertEqual(mhc["t4096_case"]["selected_regime"], "tailpipe_wsmem")
        self.assertEqual(mhc["t4096_case"]["waves_per_cta"], 5)
        self.assertEqual(mhc["t8192_case"]["selected_regime"], "tailpipe_wsmem")
        self.assertEqual(mhc["t8192_case"]["waves_per_cta"], 10)
        self.assertGreater(mhc["t8192_case"]["gmem_weight_traffic_reduction_ratio"], 0.8)
        self.assertAlmostEqual(mhc["r200_benchmarks"]["t4096_speedup_x"], 1.0188, places=3)

        sm120 = data["sm120_fp8_fa2_decode"]
        self.assertTrue(sm120["supported_on_sm120_tc"])
        self.assertEqual(sm120["unsplit_o_accum_regs_per_thread"], 128)
        self.assertEqual(sm120["vo_split_o_accum_regs_per_thread"], 32)
        self.assertEqual(sm120["o_register_reduction_factor"], 4)
        self.assertLessEqual(sm120["measured_max_abs_out_error"], 0.015)

        mla = data["rubin_mla_mtp_and_megamoe"]
        self.assertEqual(mla["graph_b64_q4_kv129"]["selected_backend"], "cute-dsl-rubin-mtp")
        self.assertEqual(mla["eager_b32_q4_kv32768"]["selected_backend"], "cute-dsl-rubin-mtp")
        self.assertEqual(mla["eager_b32_q4_kv2048"]["selected_backend"], "cute-dsl-monolithic")
        self.assertTrue(mla["graph_b64_q4_kv129"]["int32_lse_offset_overflows"])
        self.assertTrue(mla["megamoe_deferred_topk_uses_resolved_frontend_config"])

        vc = data["vc_attention_qk16"]
        self.assertEqual(vc["bf16_pv_umma_steps"], 512)
        self.assertEqual(vc["e4m3_pv_umma_steps"], 256)
        self.assertEqual(vc["mean_corr_bf16_steps"], 4)
        self.assertEqual(vc["repair_corr_bf16_steps"], 16)
        self.assertLess(vc["b300_wan22_p1_repair8_ms"], vc["b300_wan22_p1_dense_ms"])

        tl = data["tilelang_compiler_fixes"]
        self.assertEqual(tl["empty_stepped_loop"]["legacy_cpp_trip_count"], 1)
        self.assertEqual(tl["empty_stepped_loop"]["fixed_trip_count"], 0)
        self.assertEqual(tl["warp_specialized_scan"]["legacy_active_1d_count"], 0)
        self.assertEqual(tl["warp_specialized_scan"]["fixed_active_1d_count"], 32)
        self.assertEqual(tl["warp_specialized_scan"]["fixed_2d_items"], [0, 1, 2, 3])
        self.assertEqual(tl["ascend_deepgemm"]["clamped_l1_stages"], 2)

    def test_track_b_trtllm_primsts_sage_woq_vllm_lora_sglang_rust(self) -> None:
        data = self._run_and_load("trtllm-primsts-sage-vsa-sol-woq-align64-vllm-lora-sglang")
        self.assertEqual(data["phase"], 71)
        self.assertEqual(data["priority"], 80)

        sage = data["trtllm_primsts_sage_vsa_sol"]
        self.assertEqual(sage["q_scale_shape"], [16, 2049])
        self.assertEqual(sage["k_scale_shape"], [2, 129])
        self.assertEqual(sage["v_scale_shape"], [2, 2, 128])
        self.assertEqual(sage["k_summary_scale_shape"], [2, 33])
        self.assertEqual(sage["seq1_q_scale_start_slot"], 1025)
        self.assertEqual(sage["seq1_k_scale_start_slot"], 65)
        self.assertEqual(sage["blackwell_tma_oob_prefix_min_rows_bf16_h16_d128"], 32)
        self.assertGreater(sage["b200_measured_accuracy"]["int8_worst_cosine"], 0.999)

        woq = data["trtllm_int8_woq_moe_padding"]
        self.assertTrue(woq["tp2_inter1856_gated"]["legacy_unaligned_error"])
        self.assertEqual(woq["tp2_inter1856_gated"]["inter_per_partition"], 928)
        self.assertEqual(woq["tp2_inter1856_gated"]["inter_padded"], 960)
        self.assertEqual(woq["tp2_inter1856_gated"]["pad_rows_per_projection"], 32)
        self.assertTrue(woq["tp2_inter1856_gated"]["post_promotion_fc1_fc2_check_passes"])
        self.assertEqual(woq["tp1_inter200_nongated"]["inter_padded"], 256)
        self.assertEqual(woq["mx_phi3_dense_abi_id"], "trtllm-phi3-dense-target-layout-v1")

        vllm_lora = data["vllm_lora_moe_and_rocm_mxfp4"]
        self.assertFalse(vllm_lora["legacy_reload_preserves_lora_context"])
        self.assertTrue(vllm_lora["fixed_reload_preserves_lora_context"])
        self.assertTrue(vllm_lora["minimax_m3_swigluoai"]["rocm_aiter_w4a16_mxfp4_forces_interleave_layout"])

        sgl = data["sglang_rust_pd_and_muse_tools"]
        self.assertEqual(sgl["rust_pd_bootstrap_port_cases"]["http_registry_default_8998"], 31000)
        self.assertEqual(
            sgl["rust_pd_bootstrap_port_cases"]["http_registry_custom_33000"],
            "ValueError: conflicts with --port",
        )
        self.assertEqual(sgl["rust_pd_bootstrap_port_cases"]["own_bootstrap_server_default_8998"], 8998)
        self.assertEqual(sgl["rust_pd_bootstrap_port_cases"]["own_bootstrap_server_custom_33000"], 33000)
        self.assertEqual(sgl["muse_namespace_tools"]["parser_registered_names"], ["browser.search", "lookup"])

    def test_track_c_mooncake_te_torchtitan_lmcache_nixl_dynamo(self) -> None:
        data = self._run_and_load("mooncake-lasthit-session-tag-te-sm120-torchtitan-nixl-dynamo")
        self.assertEqual(data["phase"], 71)
        self.assertEqual(data["priority"], 80)

        mc = data["mooncake_probe_and_session_tag"]
        self.assertEqual(mc["normal_selection"]["leased_candidate_idx"], 2)
        self.assertEqual(mc["normal_selection"]["rechecked_candidates"], [2])
        self.assertEqual(mc["race_fallback_selection"]["leased_candidate_idx"], 0)
        self.assertEqual(mc["race_fallback_selection"]["rechecked_candidates"], [2, 0])
        self.assertEqual(mc["race_fallback_selection"]["results"][4:6], [True, False])

        te = data["te_sm120_gated_tma_and_cutedsl"]
        self.assertTrue(te["fwd_bf16_to_fp8"]["fits_sm120"])
        self.assertFalse(te["bwd_bf16_to_fp8"]["fits_sm120"])
        self.assertTrue(te["bwd_bf16_to_fp8"]["fits_sm100"])
        self.assertFalse(te["fwd_fp32_to_fp8"]["fits_sm120"])
        self.assertFalse(te["bwd_fp32_to_fp8"]["fits_sm120"])
        self.assertEqual(te["bwd_fp32_to_fp8"]["total_bytes"], 229600)

        tt = data["torchtitan_yarn_and_megatron_dsa"]
        self.assertAlmostEqual(tt["gpt_oss_yarn_scale"], 0.2266575543, places=7)
        self.assertAlmostEqual(tt["dsv3_yarn_scale"], 0.1142519529, places=7)
        self.assertAlmostEqual(tt["kl_reduction_factor"], 91.67, places=2)

        kv = data["lmcache_nixl_dynamo_kv_plane"]
        self.assertEqual(kv["lmcache_p2p_allowed_l1_tags"], ["(DRAM, LOCAL)"])
        self.assertEqual(kv["lmcache_p2p_allowed_l2_tags"], [])
        self.assertEqual(kv["nixl_failed_active_release"]["fixed_status_after_rejected_release"], "NIXL_IN_PROG")
        self.assertTrue(kv["dynamo_kvcr_hybrid_filter"]["legacy_faults_on_hybrid_kvcr"])
        self.assertFalse(kv["dynamo_kvcr_hybrid_filter"]["fixed_faults_on_hybrid_kvcr"])


if __name__ == "__main__":
    unittest.main()
