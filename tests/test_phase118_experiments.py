import json
import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase118Experiments(unittest.TestCase):
    def test_track_a_flashinfer_muse_fp8_minimax_dsa_gdn_fa4_tilelang(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "flashinfer_muse_spark_norm_fp8_gemm_minimax_dsa_gdn_fa4_tilelang_phase118"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertEqual(metrics["norm_row_alignment_bytes"], 32)
        self.assertAlmostEqual(metrics["norm_16h_max_speedup_s1"], 1.5367, places=4)
        self.assertAlmostEqual(metrics["grouped_fp8_gemm_unaligned_b200_speedup"], 4.10, places=2)
        self.assertEqual(metrics["dsa_ticket_tail_bytes"], 128)
        self.assertEqual(metrics["moe_ar_ws8_nvlink_traffic_reduction_ratio"], 0.75)
        self.assertAlmostEqual(metrics["gdn_det_train_16k_b200_speedup"], 3.105, places=3)
        self.assertEqual(metrics["fa4_sparse_mla_bwd_roundings_saved"], 2)
        self.assertEqual(metrics["tilelang_simt_invalid_atomic_shapes_rejected"], 2)

    def test_track_b_sglang_fa_vllm_pcp_mla_dcp_kvcr_trtllm_helix(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "sglang_fa_dispatch_dsv41_vllm_pcp_mla_dcp_kvcr_trtllm_helix_a2a_phase118"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertEqual(metrics["fa_bypassed_calls"], 0)
        self.assertEqual(metrics["env_override_leaks_on_exception"], 0)
        self.assertFalse(metrics["batch_sharded_draft_logits_under_pcp"])
        self.assertEqual(metrics["mla_dcp_ragged_causal_rows_after"], [2, 1, 0, 0])
        self.assertEqual(metrics["kvcr_containers_saved_per_job"], 3)
        self.assertEqual(metrics["helix_a2a_moe_compute_reduction_factor"], 8)
        self.assertEqual(metrics["kimi_k3_vision_effective_tp_at_tp16"], 1)

    def test_track_c_megatron_te_mooncake_dynamo_nixl_and_cards(self) -> None:
        script = (
            ROOT
            / "experiments"
            / "megatron_fp32_token_scale_te_mxfp8_mooncake_ep_jit_dynamo_nixl_phase118"
            / "run.py"
        )
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=True)
        metrics = json.loads(res.stdout.strip())
        self.assertGreater(metrics["megatron_token_scale_error_reduction_ratio"], 0.6)
        self.assertEqual(metrics["mooncake_ep_aot_variants_avoided"], 256)
        self.assertEqual(metrics["mooncake_batch_probe_leases_acquired"], 0)
        self.assertEqual(metrics["nixl_rocm_ipc_min_zcopy_bytes"], 0)
        self.assertEqual(metrics["nixl_xgmi_vs_host_stage_bw_gain_ratio"], 8.0)
        self.assertEqual(metrics["dynamo_lws_volcano_rbac_verbs_required"], 0)

        cards = (ROOT / "docs" / "reports" / "ai-infra-phase118-benchmark-cards.html").read_text(encoding="utf-8")
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", cards)
        for i in range(1, 8):
            self.assertIn(f"<span>{i} / 7</span>", cards)


if __name__ == "__main__":
    unittest.main()
