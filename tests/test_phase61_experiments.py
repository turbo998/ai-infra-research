#!/usr/bin/env python3
"""Unit tests for Phase 61 (priority: 70) deterministic experiments and 7-Card visual report."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPhase61Experiments(unittest.TestCase):
    def test_track_a_flashinfer_mxfp8_ar_kda_gdn_rng_tilelang(self) -> None:
        exp_dir = ROOT / "experiments" / "flashinfer-mxfp8-ar-norm-kda-grad-gdn-cp-rng-tilelang"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 61)
        self.assertEqual(summary["priority"], 70)
        self.assertTrue(summary["all_invariants_passed"])

        mxfp8 = summary["mxfp8_allreduce_rmsnorm_fusion"]
        self.assertEqual(mxfp8["patterns"]["kARResidualRMSNormMXFP8Quant"], 13)
        self.assertEqual(mxfp8["patterns"]["kARResidualRMSNormOutMXFP8Quant"], 14)
        self.assertEqual(mxfp8["swizzled_128x4_scale_bytes"], 4608)
        self.assertEqual(mxfp8["row_scale_writes_saved_pct"], 96.88)
        self.assertEqual(mxfp8["b300_tp4_minimax_m3_decode_tpot_ms"]["mnnvl_delta_pct"], -2.19)

        kda_gdn = summary["kda_chunked_grad_and_gdn_cp_sm107"]
        self.assertEqual(kda_gdn["gb300_kda_chunked_grad"][1]["bwd_speedup"], 1.892)
        self.assertEqual(kda_gdn["r200_gdn_cp_geomean_speedup_48_shapes"], 1.048)

        guards = summary["runtime_rng_and_tilelang_guards"]
        self.assertFalse(guards["ragged_auto_prefix_retention"]["stale_graph_pointer_after_replan_new"])
        self.assertEqual(guards["speculative_sampling_philox_rng"]["old_overlapping_draws"], [8])
        self.assertEqual(guards["speculative_sampling_philox_rng"]["new_overlapping_draws"], [])
        self.assertTrue(guards["tilelang_compiler_guards"]["vectorize_non_unit_step_rejected"])

    def test_track_b_trtllm_dwdp_tma_sglang_mtp_kvshard_vllm_k3(self) -> None:
        exp_dir = ROOT / "experiments" / "trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 61)
        self.assertEqual(summary["priority"], 70)
        self.assertTrue(summary["all_invariants_passed"])

        trtllm = summary["trtllm_dwdp_tma_and_gdn"]
        self.assertEqual(trtllm["dwdp_fabric_gather_tma"]["k_stages"], 14)
        self.assertEqual(trtllm["dwdp_fabric_gather_tma"]["tma_in_flight_bytes_per_block"], 212992)
        self.assertEqual(trtllm["deterministic_gdn_mixed_decode"]["decode_drift_mixed_split"], 0.0)

        sglang = summary["sglang_mtp_kvshard_and_disagg"]
        self.assertEqual(sglang["mtp_kv_shard_pool"]["shared_logical_allocator_size"], 262144)
        self.assertTrue(sglang["mtp_kv_shard_pool"]["placement_bijection_exact_128"])
        self.assertFalse(sglang["disagg_abort_ack_poisoning"]["abort_ack_sent_new_safe"])
        self.assertEqual(sglang["diffusion_ipc_and_dsv4_graph"]["new_sm90_dsv4_prefill_graph_live_seq_len"], 32768)

        vllm = summary["vllm_kimi_k3_and_dsv41"]
        self.assertEqual(vllm["kimi_k3_rocm_bf16_kda_state"]["state_memory_reduction_pct"], 50.0)
        self.assertEqual(vllm["dsv41_sparse_index_sentinel"]["new_unwritten_tail_sentinel_value"], -1)

    def test_track_c_megatron_seqpack_te_symm_ar_titan_lmcache_mooncake(self) -> None:
        exp_dir = ROOT / "experiments" / "megatron-seqpack-te-symm-ar-titan-flex16-lmcache-mooncake"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 61)
        self.assertEqual(summary["priority"], 70)
        self.assertTrue(summary["all_invariants_passed"])

        train = summary["megatron_te_and_torchtitan"]
        self.assertEqual(train["megatron_seq_packing_pp_guard"]["new_stages_with_iterator"], [0, 3])
        self.assertEqual(train["te_symmetric_allreduce_bucketed_scratch"]["scratch_reduction_pct"], 63.64)
        self.assertLessEqual(train["te_symmetric_allreduce_bucketed_scratch"]["max_padding_overhead_pct"], 25.0)
        self.assertEqual(train["torchtitan_flex_attention_and_cuspy"]["new_recompile_limit"], 16)
        self.assertEqual(train["torchtitan_flex_attention_and_cuspy"]["new_eager_fallback_graphs"], 0)

        kv = summary["lmcache_mooncake_nixl_and_dynamo"]
        self.assertIn("/cache/delete", kv["lmcache_eviction_delete_ownership"]["eviction_controller_routes"])
        self.assertEqual(kv["mooncake_store_kv_reshard_and_tenant"]["total_useful_bytes_per_layer_comp"], 8192)
        self.assertEqual(kv["nixl_completion_tracing_and_eventfd"]["new_recorded_completions"], 50)
        self.assertEqual(kv["dynamo_compact_frame_and_replica_filter"]["frame_bytes_reduction_pct"], 92.03)

    def test_phase61_bilingual_docs_and_7card_report_exist(self) -> None:
        slugs = [
            "flashinfer-mxfp8-ar-norm-kda-grad-gdn-cp-rng-tilelang",
            "trtllm-kimi-k3-dwdp-tma-gdn-sglang-kvshard-mtp-vllm",
            "megatron-seqpack-te-symm-ar-titan-flex16-lmcache-mooncake",
        ]
        for slug in slugs:
            zh = ROOT / "docs" / "components" / f"{slug}.md"
            en = ROOT / "docs" / "components" / f"{slug}.en.md"
            self.assertTrue(zh.exists(), f"Missing {zh}")
            self.assertTrue(en.exists(), f"Missing {en}")
            self.assertGreater(len(zh.read_text(encoding="utf-8")), 1500)
            self.assertGreater(len(en.read_text(encoding="utf-8")), 1500)

        card_html = ROOT / "docs" / "reports" / "ai-infra-phase61-benchmark-cards.html"
        self.assertTrue(card_html.exists())
        html_text = card_html.read_text(encoding="utf-8")
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", html_text)
        for i in range(1, 8):
            self.assertIn(f"{i} / 7", html_text)


if __name__ == "__main__":
    unittest.main()
