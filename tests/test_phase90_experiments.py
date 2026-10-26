"""Unit tests for Phase 90 (`priority: 99`) deterministic standard-library experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase90Experiments(unittest.TestCase):
    def _run_and_load(self, exp_dir: str) -> dict[str, object]:
        script = ROOT / "experiments" / exp_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("verified", res.stdout)
        summary_path = ROOT / "experiments" / exp_dir / "summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_mxfp8_rubin_sm107_tilelang(self) -> None:
        data = self._run_and_load(
            "flashinfer_mxfp8_moe_rubin_finalize_sm107_tgv_dsa_wide_tilelang_phase90"
        )
        self.assertEqual(data["phase"], 90)
        self.assertEqual(data["priority"], 99)
        self.assertEqual(data["mxfp8_cute_dsl_fused_moe"]["mxfp8_effective_bytes_per_elem"], 1.03125)
        self.assertTrue(data["rubin_unfused_deterministic_finalize"]["unfused_finalize_deterministic"])
        self.assertEqual(data["sm107_tgv_deeper_pipelines"]["sm107_added_tactics"], 18)
        self.assertEqual(data["cake_dsa_wide_kimi_tp12_moe_ar_softmax"]["moe_finalize_ar_total_programs"], 32)
        self.assertTrue(data["tilelang_sm8x_sparse_deepselect_reduce"]["sm8x_sparse_rejects_fp8_e4m3fn"])

    def test_track_b_sglang_vllm_trtllm(self) -> None:
        data = self._run_and_load(
            "sglang_weakref_hicache_vae_vllm_pd_dcp_mxfp8_trtllm_primts_phase90"
        )
        self.assertEqual(data["phase"], 90)
        self.assertEqual(data["sglang_weakref_hicache_mm_vae"]["vae_seam_io_reduction_pct"], 50.0)
        self.assertEqual(
            data["vllm_pd_cache_hits_nixl_dcp_dsv4_mxfp"]["finalized_dcp_interleave_size_before_profiling"],
            16,
        )
        self.assertEqual(
            data["trtllm_phased_init_and_primts_sparse_mla"]["sm100_sparse_mla_kv_read_reduction_x"],
            16.0,
        )

    def test_track_c_fa4_megatron_mooncake_nixl_dynamo(self) -> None:
        data = self._run_and_load(
            "fa4_sparse_mla_tma_megatron_stash_dsa_flops_mooncake_nixl_dynamo_phase90"
        )
        self.assertEqual(data["phase"], 90)
        self.assertEqual(data["fa4_sm90_tile_fit_and_sm100_tma_padding"]["fitted_tile"], [128, 64])
        self.assertAlmostEqual(
            data["megatron_paged_stash_and_dsa_flops"]["dsa_attn_flops_ratio_vs_dense"],
            0.1211,
            places=4,
        )
        self.assertEqual(data["mooncake_grouped_ha_rail_lockfree_dfs"]["batch_submission_reduction_x"], 64)
        self.assertTrue(
            data["nixl_context_span_and_dynamo_mocker_metrics"]["begin_context_span_uses_exact_context_span_id"]
        )


if __name__ == "__main__":
    unittest.main()
