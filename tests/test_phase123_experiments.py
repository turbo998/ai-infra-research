"""Unit tests for Phase 123 deterministic offline experiments."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123 import (
    run as track_a,
)
from experiments.megatron_mok_mla_verl_gate_mooncake_efa_iouring_lmcache_dynamo_nixl_phase123 import (
    run as track_c,
)
from experiments.sglang_rust_mm_mha_prefix_vllm_umbp_dsv41_tiering_p2p_trtllm_phase123 import (
    run as track_b,
)


ROOT = Path(__file__).resolve().parent.parent


class TestPhase123Experiments(unittest.TestCase):
    def test_track_a_flashinfer_flashmla_tilelang_phase123(self) -> None:
        summary = track_a.build_summary()
        track_a.verify_summary(summary)
        committed = json.loads(
            (
                ROOT
                / "experiments"
                / "flashinfer_tinygemm_kcp_cudnn_bf16fp4_mamba_h3_dsv4_flashmla_tilelang_phase123"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary, committed)
        self.assertEqual(summary["tinygemm2_and_kcp_pools"]["r200_8x4096x4096_speedup"], 1.0435)
        self.assertEqual(summary["rubin_aot_and_cudnn_prefill_cache"]["b3_saved_us"], 47.67)
        self.assertEqual(summary["flashmla_war_and_tilelang_guards"]["war_error_with_barrier"], 0.0)

    def test_track_b_sglang_vllm_trtllm_phase123(self) -> None:
        summary = track_b.build_summary()
        track_b.verify_summary(summary)
        committed = json.loads(
            (
                ROOT
                / "experiments"
                / "sglang_rust_mm_mha_prefix_vllm_umbp_dsv41_tiering_p2p_trtllm_phase123"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary, committed)
        self.assertEqual(summary["sglang_rust_mm_mha_kv_sharding"]["prefix_vram_reduction_ratio"], 8.0)
        self.assertEqual(summary["vllm_umbp_dsv41_tiering_zmq"]["uaf_corruptions_with_deferred_release"], 0)
        self.assertEqual(summary["trtllm_multifrontend_modeling_v2"]["metrics_coverage_launcher_aggregated_pct"], 100.0)

    def test_track_c_megatron_verl_mooncake_lmcache_dynamo_nixl_phase123(self) -> None:
        summary = track_c.build_summary()
        track_c.verify_summary(summary)
        committed = json.loads(
            (
                ROOT
                / "experiments"
                / "megatron_mok_mla_verl_gate_mooncake_efa_iouring_lmcache_dynamo_nixl_phase123"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(summary, committed)
        self.assertEqual(summary["megatron_mla_mok_shared_expert"]["weight_inflation_factor"], 4.0)
        self.assertEqual(summary["verl_vllm_submission_gate"]["pause_speedup"], 933.33)
        self.assertEqual(summary["mooncake_lmcache_dynamo_nixl"]["empty_notifs_after"], 0)


if __name__ == "__main__":
    unittest.main()
