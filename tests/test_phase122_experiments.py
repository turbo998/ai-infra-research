"""Unit tests for Phase 122 Track A, Track B, and Track C experiment suites."""

from __future__ import annotations

import unittest

from experiments.flashinfer_cake_fp8_gemm_sampling_gdn_mqa_nvfp4_cutlass_tilelang_phase122.run import (
    run_suite as run_track_a,
)
from experiments.megatron_refit_dp_nccl_2323_cft_te_mxfp8_fa4_mla_mooncake_lmcache_dynamo_nixl_phase122.run import (
    run_suite as run_track_c,
)
from experiments.sglang_rocm_profiler_lora_bcg_vllm_sleep_harmony_trtllm_kda_phase122.run import (
    run_suite as run_track_b,
)


class TestPhase122Experiments(unittest.TestCase):
    def test_track_a_flashinfer_cutlass_tilelang(self) -> None:
        summary = run_track_a()
        self.assertEqual(summary["phase"], 122)
        self.assertEqual(summary["priority"], 131)
        fp8 = summary["cake_grouped_fp8_gemm"]
        self.assertEqual(fp8["programs_after"], 29)
        self.assertEqual(fp8["host_routes_after"], 28)
        self.assertEqual(fp8["max_speedup_x"], 1.72)
        self.assertEqual(fp8["route_m2432_n256_on_152_sms"], "bs_ue8m0_swap_rp32_s11")
        self.assertEqual(fp8["route_m2432_n256_on_148_sms"], "existing_ue8m0_64_or_128_row")

        paged = summary["paged_auto_and_nvfp4_tma"]
        self.assertEqual(paged["legacy_plan3_q_lens_external_table"], [64, 64])
        self.assertEqual(paged["fixed_plan3_q_lens_external_table"], [16, 32])
        self.assertEqual(paged["nvfp4_oob_bytes_at_n2048_before_fix"], 1024)
        self.assertEqual(paged["nvfp4_oob_bytes_after_fix"], 0)

        samp = summary["cake_sampling_gdn_and_dense_mqa"]
        self.assertEqual(samp["sm120_stage1_builds_after"], 108)
        self.assertEqual(samp["c8_e40_max_vocab_capacity"], 163840)
        self.assertEqual(samp["gdn_bwd_prep_sm107_grid_after"], 296)
        self.assertEqual(samp["dense_mqa_blocks_scanned_per_warp_round"], 768)

        ct = summary["cutlass_48_and_tilelang_autotuner"]
        self.assertEqual(ct["rubin_sm107_tmem_columns"], 576)
        self.assertTrue(ct["tilelang_legacy_profile_tuple_collision"])
        self.assertTrue(ct["tilelang_fixed_profile_tuples_distinct"])

    def test_track_b_sglang_vllm_trtllm(self) -> None:
        summary = run_track_b()
        self.assertEqual(summary["phase"], 122)
        sgl = summary["sglang_rocm_lora_bcg"]
        self.assertEqual(sgl["rocm_gpu_trace_events_before"], 0)
        self.assertEqual(sgl["rocm_gpu_trace_events_after"], 29728)
        self.assertEqual(sgl["lora_legacy_int_alpha_0_5"], 0)
        self.assertEqual(sgl["lora_fixed_float_alpha_0_5"], 0.5)
        self.assertEqual(sgl["bcg_caller_allocated_sites_count"], 4)

        vllm = summary["vllm_sleep_and_harmony"]
        self.assertEqual(vllm["sleep_gauges_level1"]["weights_offloaded"], 1.0)
        self.assertEqual(vllm["sleep_gauges_level2"]["discard_all"], 1.0)
        self.assertEqual(vllm["harmony_reasoning_tokens"], 32)

        trt = summary["trtllm_nvfp4_moe_and_kda"]
        self.assertEqual(trt["nvfp4_moe_cutedsl_vs_trtllmgen_mismatch_pct"], 0.0)
        self.assertLessEqual(trt["kda_silu_fast_rel_error_pct_at_h_minus_2"], 1.35)
        self.assertEqual(trt["kda_b200_unit_tests_passed"], 293)

    def test_track_c_megatron_nccl_te_fa4_mooncake_lmcache_nixl(self) -> None:
        summary = run_track_c()
        self.assertEqual(summary["phase"], 122)
        mn = summary["megatron_and_nccl"]
        self.assertEqual(mn["megatron_legacy_sources_dp4"], [0, 0, 0, 0])
        self.assertEqual(mn["megatron_fixed_sources_dp4"], [0, 1, 2, 3])
        self.assertEqual(mn["nccl_gb300_allreduce_sym_2gb_gbs"]["NVL64"], 992)
        self.assertEqual(mn["nccl_gb300_allgather_ce_8gb_gbs"]["NVL8"], 751)

        tf = summary["te_and_fa4"]
        self.assertEqual(tf["te_tile_work_reduction_x"], 32.0)
        self.assertEqual(tf["fa4_overwritten_next_seq_rows_before_fix"], 64)
        self.assertEqual(tf["fa4_overwritten_next_seq_rows_after_fix"], 0)

        mln = summary["mooncake_lmcache_dynamo_nixl"]
        self.assertEqual(mln["mooncake_default_max_blocks_per_context"], 200000)
        self.assertTrue(mln["mooncake_hccs_subblock_512b_rejected"])
        self.assertEqual(mln["lmcache_rss_reduction_x"], 32.0)
        self.assertFalse(mln["nixl_restart_with_process_labels"])
        self.assertTrue(mln["nixl_collision_with_process_labels"])


if __name__ == "__main__":
    unittest.main()
