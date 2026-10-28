import json
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase117Experiments(unittest.TestCase):
    def test_track_a_flashinfer_sm107_cake_and_tilelang(self) -> None:
        script = ROOT / "experiments" / "flashinfer_xqa_sm107_cake_mxfp8_gemm_vscale_kda_tilelang_phase117" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertEqual(metrics["xqa_cvt_inst_reduction_ratio"], 0.5)
        self.assertEqual(metrics["sm107_bandwidth_fill_gain_ratio"], 2.0)
        self.assertGreater(metrics["cake_mxfp8_cta_wave_saving_ratio"], 0.6)
        self.assertGreater(metrics["cake_fp8_epilogue_stall_reduction_ratio"], 0.35)
        self.assertTrue(metrics["all_glm53_tp_covered"])
        self.assertEqual(metrics["fallback_wait_group_immediate"], 0)

    def test_track_b_sglang_mtp_dspark_eplb_and_vllm_tma(self) -> None:
        script = ROOT / "experiments" / "sglang_vattn_dspark_engram_moe_eplb_vllm_tma_kda_phase117" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertEqual(metrics["gfx950_cu_utilization_gain_ratio"], 8.0)
        self.assertEqual(metrics["dspark_step_overhead_saving_us"], 68.0)
        self.assertEqual(metrics["eplb_decode_graph_node_reduction"], 2)
        self.assertGreater(metrics["gdn_mxfp4_weight_compression_ratio"], 3.7)
        self.assertAlmostEqual(metrics["vllm_glm53_tma_e2e_gain_pct"], 3.3, places=2)
        self.assertEqual(metrics["reserved_kv_budget_overrun_mib"], 0.0)

    def test_track_c_te_megatron_lmcache_dynamo_nixl_and_cards(self) -> None:
        script = ROOT / "experiments" / "te_recipe_cutedsl_megatron_glu_mooncake_lmcache_dynamo_nixl_phase117" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertAlmostEqual(metrics["te_step_fuser_cpu_saved_us"], 475.8, places=1)
        self.assertGreater(metrics["cutedsl_launch_guard_reduction_ratio"], 0.84)
        self.assertEqual(metrics["megatron_glu_checkpoint_max_abs_err"], 0.0)
        self.assertEqual(metrics["lmcache_s3_silent_buffer_corruptions"], 0)
        self.assertEqual(metrics["dynamo_accidental_foreign_deletions"], 0)
        self.assertEqual(metrics["nixl_degraded_tracing_failed_transfers"], 0)

        cards = (ROOT / "docs" / "reports" / "ai-infra-phase117-benchmark-cards.html").read_text(encoding="utf-8")
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", cards)
        for i in range(1, 8):
            self.assertIn(f"<span>{i} / 7</span>", cards)


if __name__ == "__main__":
    unittest.main()
