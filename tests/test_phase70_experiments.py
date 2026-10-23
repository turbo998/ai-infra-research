import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase70Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        script = REPO_ROOT / "experiments" / rel_dir / "run.py"
        summary_file = REPO_ROOT / "experiments" / rel_dir / "summary.json"
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        return json.loads(summary_file.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_rubin_megamoe_qsa_rmsnorm_tilelang(self) -> None:
        data = self._run_and_load("flashinfer-rubin-megamoe-auto-qsa-wide-rmsnorm-pdl-tilelang")
        self.assertEqual(data["phase"], 70)
        self.assertEqual(data["priority"], 79)

        megamoe = data["rubin_megamoe_auto"]
        self.assertEqual(
            megamoe["backend_revision"],
            "sm107-block-scaled-1667b47a-runtime-options-v1",
        )
        self.assertTrue(megamoe["ep_homogeneous_case"]["collective_replace"])
        self.assertTrue(megamoe["ep_homogeneous_case"]["uncoordinated_deadlock_risk"])
        self.assertTrue(megamoe["ep_homogeneous_case"]["preserved_in_captured_list"])
        self.assertTrue(megamoe["ep_mixed_autotune_case"]["mixed_autotune_error"])
        self.assertFalse(megamoe["ep_capture_stream_case"]["effective_tuning_enabled"])

        qsa = data["cake_qsa_nvfp4_sm80_wide"]
        self.assertEqual(qsa["rows_128_case"]["selected_program"], "decode_whole_tile")
        self.assertEqual(qsa["rows_128_case"]["tiles_per_cta"], 17)
        self.assertEqual(qsa["rows_64_case"]["selected_program"], "decode")
        self.assertEqual(qsa["rows_64_case"]["tiles_per_cta"], 9)
        self.assertAlmostEqual(qsa["a100_benchmarks"]["a100_80gb_nhd_speedup_x"], 1.0852, places=3)

        pdl = data["cake_rmsnorm_train_pdl"]
        self.assertEqual(pdl["gb300_sm_count"], 152)
        self.assertEqual(pdl["b200_sm_count"], 148)
        self.assertLessEqual(pdl["non_cooperative_pdl_max_slowdown_x"], 1.13)

        ws = data["trtllm_moe_workspace"]
        self.assertEqual(ws["unshuffled_ds_fp8_all_tiles"]["legacy_measured_bytes"], 0)
        self.assertGreater(ws["unshuffled_ds_fp8_all_tiles"]["fixed_measured_bytes"], 0)
        self.assertLess(
            ws["unshuffled_ds_fp8_tactic_16_0"]["fixed_measured_bytes"],
            ws["unshuffled_ds_fp8_all_tiles"]["fixed_measured_bytes"],
        )

        tl_math = data["tilelang_16bit_math"]["nextafter_subnormal_cases"]
        self.assertEqual(tl_math["pos_zero_to_neg_one"], "0x8001")
        self.assertEqual(tl_math["neg_zero_to_pos_one"], "0x1")
        self.assertEqual(tl_math["max_subnormal_to_min_normal"], "0x80")
        self.assertEqual(tl_math["min_normal_to_max_subnormal"], "0x7f")

    def test_track_b_vllm_moe_permute_rdna_sglang_inkling_trtllm(self) -> None:
        data = self._run_and_load("vllm-moe-permute-rdna-w4a16-sglang-inkling-trtllm")
        self.assertEqual(data["phase"], 70)
        self.assertEqual(data["priority"], 79)

        perm = data["vllm_moe_permute_invalid_ids"]
        self.assertEqual(perm["legacy_first_sorted_key"], -1)
        self.assertEqual(perm["fixed_first_sorted_key"], 0)
        self.assertEqual(perm["fixed_tail_sentinel_count"], 8)
        self.assertEqual(perm["num_valid_expanded_rows"], 12)
        self.assertGreater(perm["scale_inflation_ratio"], 50.0)

        rdna = data["vllm_rdna_w4a16_packed_fp16"]
        self.assertEqual(rdna["symmetric_case"]["packed_fp16x2_hex"], "0x640E6403")
        self.assertEqual(rdna["symmetric_case"]["max_abs_error"], 0.0)
        self.assertEqual(rdna["asymmetric_case"]["max_abs_error"], 0.0)
        self.assertTrue(rdna["gfx942_kimi_k3_int4_requant_supported"])
        self.assertTrue(rdna["spec_decode_drafter_lora_allowed"])

        inkling = data["sglang_inkling_lm_head_output_fp32"]
        self.assertEqual(inkling["fp32_output_argmax"], 1)
        self.assertEqual(inkling["bf16_output_argmax"], 0)
        self.assertTrue(inkling["greedy_winner_flipped_in_bf16"])

        quark = data["sglang_quark_block_fp8"]
        self.assertTrue(quark["resolved_quark_qkv_scale"].endswith("weight_scale_inv"))
        self.assertTrue(quark["preserved_mxfp4_expert_scale"].endswith("w13_weight_scale"))
        self.assertEqual(quark["resolved_layer_weight_block_size"], [64, 128])

    def test_track_c_megatron_te_torchtitan_lmcache_mooncake_nixl_dynamo(self) -> None:
        data = self._run_and_load("megatron-mhc-recompute-stream-slice-te-mxfp8-lmcache-mooncake")
        self.assertEqual(data["phase"], 70)
        self.assertEqual(data["priority"], 79)

        meg = data["megatron_recompute_and_slice"]
        self.assertEqual(
            meg["uniform_plan_without_atomic"],
            [False, True, False, True, False, True],
        )
        self.assertEqual(
            meg["uniform_plan_with_atomic_1_2"],
            [False, False, True, False, True, True],
        )
        self.assertEqual(len(meg["sliced_merged_segments"]), 3)
        self.assertEqual(
            meg["sliced_merged_segments"][1],
            [[3, 4, "dense"], [4, 6, "moe"]],
        )

        te = data["te_mxfp8_register_resident"]
        self.assertEqual(te["code_254_legacy_hex"], "0x00000000")
        self.assertEqual(te["code_254_fixed_hex"], "0x00400040")
        self.assertEqual(te["code_255_legacy_hex"], "0xFF7FFF80")
        self.assertEqual(te["code_255_fixed_hex"], "0x7FFF7FFF")
        self.assertLess(te["bidim_geomean_after_vs_tma"], te["bidim_geomean_before_vs_tma"])

        tt = data["torchtitan_rl_routing_and_verifiers"]
        self.assertTrue(tt["case_6_2_2_2"]["overloaded"])
        self.assertEqual(tt["case_6_2_2_2"]["chosen_idx"], 1)
        self.assertFalse(tt["case_4_3_3_2"]["overloaded"])
        self.assertEqual(tt["case_4_3_3_2"]["chosen_idx"], 0)
        self.assertTrue(tt["case_1_0_0_0"]["overloaded"])
        self.assertEqual(tt["verifiers_040_timeout_with_ok_true_status"], "ERROR_TIMEOUT")

        kv = data["kv_storage_and_serving_plane"]
        self.assertEqual(kv["lmcache_nixl_pool"]["reported_pool_size_slots"], 64)
        self.assertTrue(kv["lmcache_nixl_pool"]["invariant_free_le_pool_size"])
        self.assertEqual(kv["mooncake_io_uring_and_ascend"]["transfer_status"], "FAILED")
        self.assertTrue(kv["nixl_and_dynamo"]["nixl_ucx_cross_process_trace_verified"])


if __name__ == "__main__":
    unittest.main()
