import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase72Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        script = REPO_ROOT / "experiments" / rel_dir / "run.py"
        summary_file = REPO_ROOT / "experiments" / rel_dir / "summary.json"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        return json.loads(summary_file.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_sm100_qsa_nvfp4_rmsnorm_mhc_frost_dsa_tilelang(self) -> None:
        data = self._run_and_load("flashinfer-sm100-qsa-nvfp4-rmsnorm-mhc-frost-dsa-tilelang")
        self.assertEqual(data["phase"], 72)
        self.assertEqual(data["priority"], 81)

        qsa = data["qsa_nvfp4"]
        self.assertEqual(qsa["tiles_of_64"], 33)
        self.assertEqual(
            list(qsa["b200_sm100a_148sm_splits"].values()),
            [33, 33, 18, 9, 4, 2, 1, 1],
        )
        self.assertTrue(qsa["grace_gb300_eager_rows4_coresident"])
        self.assertFalse(qsa["grace_gb300_graph_rows4_coresident"])
        self.assertFalse(qsa["x86_b200_eager_rows4_coresident"])
        self.assertTrue(qsa["e4m3_rebias_fold_exact"])
        self.assertGreater(qsa["upstream_benchmarks"]["b200_eager_memset_hnd_rows128_speedup_x"], 2.9)

        rms = data["rmsnorm_and_mhc"]["rmsnorm_b200_t65536"]
        self.assertEqual(rms["legacy_chunk_rows"], 28)
        self.assertEqual(rms["new_chunk_rows"], 37)
        self.assertTrue(rms["pdl_store_trigger_t2048"])
        self.assertFalse(rms["pdl_store_trigger_t8192"])
        self.assertEqual(rms["sm107a_min_cuda"], "13.4")
        self.assertEqual(rms["sm107a_updated_entry_count"], 41)

        mhc = data["rmsnorm_and_mhc"]["mhc_train_h4096"]
        self.assertEqual(mhc["saved_stats_bytes_per_token"], 256)
        self.assertAlmostEqual(mhc["bf16x2_unpack_lo"], -2.5, places=5)
        self.assertAlmostEqual(mhc["bf16x2_unpack_hi"], 1.75, places=5)

        frost = data["frost_and_dsa"]["frost_low_latency"]
        self.assertEqual(frost["w4a16_t4_route_slots"], 80)
        self.assertEqual(frost["w4a16_t4_scratch_slots"], 640)
        self.assertEqual(frost["w4a16_t16_scratch_slots"], 1280)

        dsa = data["frost_and_dsa"]["dsa_indexer_sm100"]
        self.assertEqual(dsa["sampled_cand_cap_floor"], 16384)
        self.assertEqual(dsa["unsampled_cand_cap_floor"], 8192)

        tl = data["tilelang_fixes"]
        self.assertTrue(tl["vectorize_loop_step_guard"]["step_2_rejected"])
        self.assertEqual(tl["atomic_signed_int64"]["legacy_unsigned_cast_atomic_max"], -5)
        self.assertEqual(tl["atomic_signed_int64"]["fixed_signed_atomic_ref_max"], 2)
        self.assertTrue(tl["nvrtc_from_database_sets_adapter_libpath"])

    def test_track_b_trtllm_dsv4_rubin_mxfp8_vllm_mla_spec_sglang_hicache(self) -> None:
        data = self._run_and_load("trtllm-dsv4-rubin-mxfp8-vllm-mla-spec-sglang-hicache")
        self.assertEqual(data["phase"], 72)
        self.assertEqual(data["priority"], 81)

        dsv4 = data["trtllm_rubin_and_moe"]["dsv4_rubin_sm107"]
        self.assertTrue(dsv4["use_cute_dsl_blockscaling_mm_default_on_sm107"])
        self.assertTrue(dsv4["q_rope_applied_when_q_pe_omitted"])
        self.assertTrue(dsv4["o_b_proj_allow_unpartitioned_output"])
        self.assertEqual(dsv4["legacy_extra_bytes_per_token"], 79872)
        self.assertEqual(dsv4["fused_rubin_extra_bytes_per_token"], 0)
        self.assertAlmostEqual(dsv4["cutedsl_dep4_gsm8k_ref_al"], 4.433448, places=5)

        async_q = data["trtllm_rubin_and_moe"]["moe_async_input_quant"]["decision_matrix"]
        self.assertFalse(async_q["eager_single_stream"])
        self.assertTrue(async_q["cuda_graph_multi_stream"])
        self.assertTrue(async_q["cuda_graph_capturing"])
        self.assertFalse(async_q["with_comm_strategy"])
        self.assertFalse(async_q["with_router_weight_on_input"])

        awq = data["trtllm_rubin_and_moe"]["nvfp4_awq_pre_quant_scale"]
        self.assertAlmostEqual(awq["true_scaled_dot"], 12.0, places=5)
        self.assertAlmostEqual(awq["buggy_unscaled_dot"], 40.0, places=5)

        gemma4 = data["trtllm_rubin_and_moe"]["gemma4_mm_and_lora"]
        self.assertEqual(gemma4["gemma4_padded_patches_per_frame"], 2520)
        self.assertTrue(gemma4["lora_reload_uses_request_ckpt_source"])

        vs = data["vllm_and_sglang"]
        self.assertEqual(
            vs["vllm_batch_sharded_adaptive_verification"]["tp2_rank0_logits_scheduled_vs_admitted"],
            [8, 5],
        )
        self.assertEqual(
            vs["vllm_modelopt_dflash_mapping"]["mapped_keys"],
            [
                "model.layers.1.mlp.gate_proj",
                "model.layers.1.self_attn.q_proj",
                "model.layers.2.self_attn.q_proj",
            ],
        )
        self.assertEqual(vs["sglang_hicache_and_diffusion"]["fixed_rematch_max_prefix_len"], 17)
        self.assertEqual(vs["sglang_hicache_and_diffusion"]["video_diffusion_per_step_host_syncs_after_fix"], 0)

    def test_track_c_mooncake_lmcache_torchtitan_te_nixl_dynamo(self) -> None:
        data = self._run_and_load("mooncake-hca-numa-lmcache-packed-torchtitan-ulysses-te-nixl-dynamo")
        self.assertEqual(data["phase"], 72)
        self.assertEqual(data["priority"], 81)

        mc = data["mooncake"]
        self.assertEqual(mc["hca_numa_tier0_selection"]["selected_segment"], "seg_numa0_mlx5_0")
        self.assertEqual(mc["hca_numa_tier0_selection"]["selected_tier"], 0)
        self.assertEqual(mc["hca_numa_tier1_fallback"]["selected_segment"], "seg_legacy_unknown")
        self.assertEqual(mc["hca_numa_tier1_fallback"]["selected_tier"], 1)
        self.assertEqual(mc["rdma_post_send_clock_reads_per_32wr"]["reduction_factor"], 32)
        self.assertTrue(mc["memfabric_batch_register"]["buf0_same_2mb_npu_page_as_buf1"])

        lmc = data["lmcache"]
        self.assertGreater(lmc["packed_token_ledger_16k_256steps"]["copy_reduction_factor"], 250.0)
        self.assertEqual(lmc["hybrid_kv_window_chunks_c576"]["sliding_window_2048"], 4)
        self.assertEqual(lmc["hybrid_kv_window_chunks_c576"]["sliding_window_2048_with_full_sw_kv"], -1)
        self.assertEqual(lmc["hybrid_kv_window_chunks_c576"]["gdn_recurrent_with_full_sw_kv"], 1)

        tt = data["torchtitan_and_te"]["torchtitan_muse_spark_816b_gb300"]
        self.assertEqual(tt["new_trainer_gpus"], 128)
        self.assertEqual(tt["fsdp_all_gather_group_reduction_x"], 8)
        self.assertEqual(tt["tokens_per_cp_rank"], 16384)

        te = data["torchtitan_and_te"]["te_grouped_tensor_and_perm"]
        self.assertFalse(te["wrapper_subclass_internal_requires_grad"])
        self.assertEqual(te["capacity_drop_test"]["kept_count"], 5)
        self.assertEqual(te["capacity_drop_test"]["capped_row_ids"][5:], [-1, -1, -1])

        nixl = data["nixl_and_dynamo"]["nixl_ucx_treq_tag"]
        self.assertTrue(nixl["matches_upstream_test_vector"])
        self.assertEqual(nixl["payload_bytes"], 16)


if __name__ == "__main__":
    unittest.main()
