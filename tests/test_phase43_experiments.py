"""Unit tests for Phase 43 (`priority: 51`) deterministic experiments and visual report."""

from __future__ import annotations

from pathlib import Path
import unittest

from experiments.flashinfer_cake_gdp_nvfp4_4over6_conv1d_bwd.run import (
    build_summary as build_flashinfer_p42,
    evaluate_nvfp4_block_4over6,
    route_cake_causal_conv1d_bwd,
    route_cake_gdp_module,
)
from experiments.megatron_dsa_gqa_gdn_mooncake_store_reshard_quota.run import (
    build_gdp_chunk_descriptors,
    build_summary as build_megatron_mooncake_p42,
    simulate_mooncake_tenant_quota_lifecycle,
)
from experiments.trtllm_dkg_dsa_mqa_kv_contract_vllm_watchdog.run import (
    atom_ctx_offsets,
    build_summary as build_trtllm_vllm_p42,
    pick_dsl_split,
)


class Phase42ExperimentsTest(unittest.TestCase):
    def test_flashinfer_cake_gdp_nvfp4_4over6_and_conv1d_bwd(self) -> None:
        summary = build_flashinfer_p42()
        gdp = summary["cake_gdp"]
        self.assertTrue(gdp["readonly_matches_active_output"])
        self.assertTrue(gdp["readonly_preserves_initial_state"])
        self.assertTrue(gdp["active_mutates_state"])
        self.assertEqual(gdp["gb200_benchmark"]["shapes_won"], 48)
        self.assertEqual(gdp["gb300_benchmark"]["shapes_won"], 47)

        r_small = route_cake_gdp_module("sm_100a", batch_size=2, heads=80, num_tokens=1, num_householder=3)
        self.assertEqual(r_small["body"], "splitv")
        self.assertEqual(r_small["grid_x"], 1280)

        r_large = route_cake_gdp_module("sm_103a", batch_size=256, heads=10, num_tokens=8, num_householder=3)
        self.assertEqual(r_large["body"], "lanerow")
        self.assertEqual(r_large["variant"], "l8r2")

        nvfp4 = summary["nvfp4_slot_append"]
        self.assertEqual(nvfp4["clustered_block_eval"]["selected_policy"], "4over6")
        self.assertEqual(nvfp4["uniform_block_eval"]["selected_policy"], "standard_6")
        self.assertEqual(nvfp4["sm80_eager_prefill_sync_removal"]["saved_ms"], 47.7)

        bwd_sm103 = route_cake_causal_conv1d_bwd("sm_103a", 512)
        self.assertTrue(bwd_sm103["fold_overlap"])
        self.assertEqual(bwd_sm103["smem_total_bytes"], 66560)
        bwd_sm107_wide = route_cake_causal_conv1d_bwd("sm_107a", 2048)
        self.assertTrue(bwd_sm107_wide["roll_prefetch"])
        self.assertEqual(bwd_sm107_wide["smem_total_bytes"], 82944)

    def test_trtllm_cute_dsl_uneven_split_kv_contract_and_vllm_watchdog(self) -> None:
        summary = build_trtllm_vllm_p42()
        self.assertEqual(atom_ctx_offsets(5, 2), [2, 0])
        self.assertEqual(atom_ctx_offsets(7, 3), [4, 2, 0])

        split7 = pick_dsl_split(next_n=7, num_sms=148, batch_size=32, max_ctx=16384)
        self.assertEqual(split7["mode"], "in_kernel_uneven_split")
        self.assertEqual(split7["num_kernel_atoms"], 3)
        self.assertEqual(split7["kv_read_reduction_pct"], 57.14)

        kv = summary["kv_transfer_backend_contract"]
        self.assertFalse(kv["unquiesced_attempt"]["undelivered_memory_reusable"])
        self.assertTrue(kv["quiesced_attempt"]["undelivered_memory_reusable"])
        self.assertTrue(kv["quiesced_attempt"]["mpi_teardown"]["shutdown_ack_received"])

        wd = summary["vllm_threaded_watchdog"]
        self.assertEqual(wd["dump_count"], 2)
        self.assertEqual(wd["dumps"][0]["timestamp"], 910.0)
        self.assertEqual(wd["dumps"][1]["timestamp"], 1320.0)

    def test_megatron_dsa_gqa_gdn_mooncake_and_nixl(self) -> None:
        summary = build_megatron_mooncake_p42()
        gdp_desc = build_gdp_chunk_descriptors([0, 110, 174, 310], chunk_size=64, padded_chunk_capacity=12)
        self.assertEqual(gdp_desc["active_chunk_count"], 6)
        self.assertEqual(gdp_desc["padding_slots"], 6)

        reshard = summary["mooncake_store_reshard_and_quota"]["reshard_plan"]
        self.assertEqual(reshard["upload_writer_count"], 2)
        self.assertEqual(reshard["load_range_count"], 32)

        quota = simulate_mooncake_tenant_quota_lifecycle(
            quota_bytes=1000,
            initial_write_bytes=400,
            upsert_reserved_bytes=850,
            upsert_actual_bytes=900,
        )
        self.assertEqual(quota["charged_during_processing_upsert"], 1300)
        self.assertEqual(quota["charged_after_upsert_settle"], 900)
        self.assertEqual(quota["evict_target_bytes"], 200)

        nixl = summary["nixl_xfer_telemetry"]
        self.assertEqual(nixl["zero_redraws"], 2)
        self.assertEqual(nixl["final_xfer_execution"], 2)

    def test_phase42_benchmark_cards_html_contract(self) -> None:
        report_path = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "reports"
            / "ai-infra-phase43-benchmark-cards.html"
        )
        self.assertTrue(report_path.exists())
        html = report_path.read_text(encoding="utf-8")
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", html)
        for badge in ("1 / 7", "2 / 7", "3 / 7", "4 / 7", "5 / 7", "6 / 7", "7 / 7"):
            self.assertIn(badge, html)


if __name__ == "__main__":
    unittest.main()
