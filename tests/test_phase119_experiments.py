import json
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase119Experiments(unittest.TestCase):
    def test_track_a_flashinfer_ssd_mxfp8_tinygemm2_kimi_vsa_qsa_kda_tilelang(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "flashinfer_ssd_combined_gated_act_mxfp8_tinygemm2_kimi_vsa_qsa_kda_tilelang_phase119"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertEqual(metrics["ssd_sm107a_regenerated_programs"], 17)
        self.assertAlmostEqual(metrics["ssd_all_93_geomean_speedup"], 1.0487, places=4)
        self.assertEqual(metrics["mxfp8_total_mk_ranges"], 8)
        self.assertEqual(metrics["mxfp8_sm103a_bwd_rowwise_new_spill_bytes"], 0)
        self.assertEqual(metrics["tinygemm2_l2_promotion_bytes"], 256)
        self.assertEqual(metrics["kimi_gb300_capacity_c"], 592)
        self.assertEqual(metrics["kda_sm107a_ctas_per_sm"], 2)
        self.assertTrue(metrics["ascend_reg_phase_constraint_satisfied"])
        self.assertTrue(metrics["bisheng_builds_separated"])
        self.assertTrue(metrics["bisheng_relocation_invariant"])

    def test_track_b_sglang_diffusion_sp_dspark_megamoe_vllm_rl_hisparse_trtllm_gvr(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "sglang_diffusion_sp_dspark_megamoe_moss_lru_vllm_rl_nccl_hisparse_trtllm_gvr_phase119"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertFalse(metrics["moss_sp_did_shard_latents_before_rope"])
        self.assertEqual(metrics["dsv4_extend_start_loc"], [0, 4, 11, 14])
        self.assertEqual(metrics["megamoe_budget_restoration_ratio"], 2.0)
        self.assertEqual(metrics["moss_high_quality_gemms_per_block"], 1)
        self.assertEqual(metrics["lru_touch_first_item"], "boot_3")
        self.assertEqual(metrics["nccl_world_size"], 12)
        self.assertTrue(metrics["lora_ep_round_robin_rejected"])
        self.assertEqual(metrics["online_acceptance_block_r"], 256)
        self.assertTrue(metrics["hisparse_oob_prevented"])
        self.assertEqual(metrics["gvr_hint_slots"], 2)
        self.assertEqual(metrics["gvr_candidates_per_group"], 32)

    def test_track_c_megatron_mhc_pinned_te_mooncake_lmcache_dynamo_nixl_and_cards(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "megatron_mhc_recompute_pinned_offload_te_fa4_perm_mooncake_lmcache_dynamo_nixl_phase119"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertEqual(metrics["pinned_alignment_bytes"], 256)
        self.assertGreater(metrics["pinned_host_memory_savings_ratio"], 0.35)
        self.assertTrue(metrics["thd_padding_mask_exact"])
        self.assertAlmostEqual(metrics["te_perm_memory_reduction_factor"], 17.14, places=2)
        self.assertTrue(metrics["mooncake_exact_elements_reconstructed"])
        self.assertEqual(metrics["dsmi_resolved_logic_ids"], [2, 3])
        self.assertEqual(metrics["buggy_stored_end"], 192)
        self.assertEqual(metrics["fixed_stored_end"], 256)
        self.assertEqual(metrics["blake3_total_threads_w8"], 16)
        self.assertEqual(metrics["nixl_plugin_api_version"], 3)

        cards = (ROOT / "docs" / "reports" / "ai-infra-phase119-benchmark-cards.html").read_text(encoding="utf-8")
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", cards)
        for i in range(1, 8):
            self.assertIn(f"<span>{i} / 7</span>", cards)


if __name__ == "__main__":
    unittest.main()
