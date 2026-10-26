from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase92Experiments(unittest.TestCase):
    def _run_and_verify(self, exp_name: str) -> dict:
        exp_dir = REPO_ROOT / "experiments" / exp_name
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        self.assertTrue(run_py.exists(), f"Missing {run_py}")

        res = subprocess.run(
            [sys.executable, str(run_py), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, f"--check failed for {exp_name}: {res.stderr or res.stdout}")
        return json.loads(summary_json.read_text(encoding="utf-8"))

    def test_track1_flashinfer_moe_gdn_sm12x_fp4_tilelang_phase92(self) -> None:
        data = self._run_and_verify("flashinfer_moe_gdn_sm12x_fp4_tilelang_phase92")
        self.assertEqual(data["phase"], 92)
        self.assertEqual(data["priority"], 101)
        self.assertEqual(data["sm12x_gdn_triangular_mma_skip"]["skipped_below_diagonal_tiles"], 12)
        self.assertEqual(data["sm12x_gdn_triangular_mma_skip"]["executed_tiles"], 20)
        self.assertAlmostEqual(data["sm12x_gdn_triangular_mma_skip"]["mma_reduction_pct"], 37.5)
        self.assertEqual(data["sm100_gdn_state_stride_divisibility"]["divisibility_with_assume"], 128)
        self.assertEqual(data["sm120_fp16_w4a16_e4m3_subnormals"]["positive_subnormal_count"], 7)
        self.assertTrue(data["sm120_fp16_w4a16_e4m3_subnormals"]["all_seven_preserved"])
        self.assertEqual(data["cudnn_rectangular_pagetable_staging"]["fast_path_copy_launches"], 1)
        self.assertEqual(data["tilelang_atomics_and_vectorize"]["fixed_signed_t1_atomic_max"], 19)
        self.assertEqual(data["tilelang_atomics_and_vectorize"]["fixed_signed_t1_atomic_min"], -12)

    def test_track2_sglang_minimax_vllm_mrv2_kv_offload_trtllm_phase92(self) -> None:
        data = self._run_and_verify("sglang_minimax_vllm_mrv2_kv_offload_trtllm_phase92")
        self.assertEqual(data["phase"], 92)
        self.assertEqual(data["priority"], 101)
        self.assertEqual(data["sglang_minimax_m31_flash_and_dspark"]["cos_sin_half_width"], 32)
        self.assertTrue(data["sglang_amd_aiter_softcap"]["max_logit_bounded_below_cap"])
        self.assertTrue(data["sglang_diffusion_cleanup_and_dp_routing"]["all_releases_routed_to_owner"])
        self.assertFalse(data["vllm_mrv2_cudagraph_profile_only_skip"]["profile_only_runs_adaptive_verification_fixed"])
        self.assertEqual(data["vllm_rocm_dsv41_krdy_barrier"]["missing_chunks_recovered"], 24)
        self.assertEqual(data["vllm_kv_offload_reset_and_nixl"]["stale_pre_reset_promotions_after_fix"], 0)
        self.assertTrue(data["trtllm_unified_kv_and_beam_histories"]["indexer_isolated_from_kv_pool"])
        self.assertTrue(data["trtllm_unified_kv_and_beam_histories"]["slot_vs_hypothesis_diverged_under_beam_swap"])

    def test_track3_te_nvfp4_deepgemm_torchtitan_mooncake_nixl_dynamo_phase92(self) -> None:
        data = self._run_and_verify("te_nvfp4_deepgemm_torchtitan_mooncake_nixl_dynamo_phase92")
        self.assertEqual(data["phase"], 92)
        self.assertEqual(data["priority"], 101)
        self.assertTrue(data["te_grouped_mlp_distributed_bf16_nvfp4"]["gather_before_quantize_for_distributed_bf16_shards"])
        self.assertTrue(data["deepgemm_megamoe_and_tensormap_fences"]["megamoe_metadata_overwrite_race_eliminated"])
        self.assertEqual(data["torchtitan_flux_shift_and_local_compile"]["seq_len_overestimate_factor"], 4)
        self.assertEqual(data["torchtitan_flux_shift_and_local_compile"]["deepseek_v4_rope_recompile_limit"], 32)
        self.assertEqual(data["mooncake_xpu_ep_and_store_guard"]["uninitialized_client_null_deref_crashes"], 0)
        self.assertEqual(data["nixl_repost_spans_and_dynamo_runtime"]["unique_post_span_ids"], 3)
        self.assertTrue(data["nixl_repost_spans_and_dynamo_runtime"]["all_posts_share_trace_id"])


if __name__ == "__main__":
    unittest.main()
