"""Unit tests for Phase 120 Track A, Track B, and Track C experiments."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase120Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        exp_dir = REPO_ROOT / "experiments" / rel_dir
        run_py = exp_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(run_py), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, msg=res.stderr or res.stdout)
        summary_path = exp_dir / "summary.json"
        self.assertTrue(summary_path.is_file())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_gb300_cake_fmha_rubin_svdquant_tilelang(self) -> None:
        s = self._run_and_load(
            "flashinfer_gb300_sf_ring_cake_fmha_rubin_mla_svdquant_tilelang_phase120"
        )
        self.assertEqual(s["phase"], 120)
        self.assertEqual(s["gb300_sf_ring_fp4_gemm"]["sf_ring_arch"], "sm_103a")
        self.assertEqual(s["gb300_sf_ring_fp4_gemm"]["num_tma_pipe_stages"], 6)
        self.assertEqual(s["gb300_sf_ring_fp4_gemm"]["num_sf_pipe_stages"], 5)
        self.assertAlmostEqual(s["gb300_sf_ring_fp4_gemm"]["max_gemm_speedup"], 1.1061, places=4)
        self.assertEqual(s["cake_fmha_paged_decode"]["total_aot_programs"], 27)
        self.assertAlmostEqual(
            s["cake_fmha_paged_decode"]["bf16_b4_kv1024_speedup"], 1.750, places=3
        )
        self.assertEqual(s["rubin_mla_and_svdquant"]["rubin_new_pdl_max_items"], 54)
        self.assertTrue(s["rubin_mla_and_svdquant"]["svdquant_single_wave_fit"])
        self.assertEqual(s["b12x_qsa_tilelang_rocm"]["b12x_bucketed_compile_count"], 6)
        self.assertTrue(s["b12x_qsa_tilelang_rocm"]["tilelang_rocm_uses_current_device"])

    def test_track_b_sglang_sm107_engram_mxfp4_vllm_hisparse_kvcr_trtllm(self) -> None:
        s = self._run_and_load(
            "sglang_sm107_2shot_engram_dp_mxfp4_vllm_hisparse_kvcr_trtllm_phase120"
        )
        self.assertEqual(s["phase"], 120)
        self.assertEqual(s["sm107_2shot_allreduce"]["arch"], "sm_107")
        self.assertEqual(
            s["sm107_2shot_allreduce"]["configs"]["tp8"]["two_shot_push_threshold_bytes"],
            16 * 1024 * 1024,
        )
        self.assertEqual(
            s["dsv41_engram_dp_collective"]["divisible_tp_sizes"],
            [2, 3, 4, 6, 8, 12],
        )
        self.assertEqual(s["sglang_moe_and_dsv4_fixes"]["mxfp4_marlin_scale_type"], "kFE8M0fnu")
        self.assertFalse(s["sglang_moe_and_dsv4_fixes"]["mxfp4_dequant_skip_flop"])
        self.assertTrue(s["vllm_and_trtllm_fixes"]["spec_lora_mapping_matches_logits_rows"])
        self.assertEqual(
            s["vllm_and_trtllm_fixes"]["hisparse_residency_shape_order"],
            ["resident_groups", "max_num_reqs", "max_num_pages"],
        )
        self.assertEqual(s["vllm_and_trtllm_fixes"]["kvcr_processed_tokens"], 112)

    def test_track_c_megatron_te_mooncake_lmcache_dynamo_nixl(self) -> None:
        s = self._run_and_load(
            "megatron_vit_rope_fp32_nccl_nvls_te_zerotensor_fa4_mooncake_lmcache_dynamo_nixl_phase120"
        )
        self.assertEqual(s["phase"], 120)
        self.assertTrue(s["megatron_vit_and_nvls"]["vit_rope_kept_fp32_outside_buffer"])
        self.assertTrue(s["megatron_vit_and_nvls"]["ring_nvls_on_blackwell_sm100"])
        self.assertFalse(s["megatron_vit_and_nvls"]["ring_nvls_on_hopper_sm90"])
        self.assertFalse(s["megatron_vit_and_nvls"]["nvls_alone_on_blackwell_sm100"])
        self.assertFalse(s["te_zerotensor_and_fa4"]["sm120_det_bwd_fa4_enabled"])
        self.assertTrue(s["te_zerotensor_and_fa4"]["sm120_det_fwd_only_fa4_enabled"])
        self.assertEqual(
            s["mooncake_efa_hixl_peermem"]["gb300_open_kmod_unset_env_path"],
            "dma_buf_export",
        )
        self.assertEqual(
            s["lmcache_dynamo_nixl"]["dynamo_token_string_allocs_without_logprobs"], 0
        )
        self.assertEqual(s["lmcache_dynamo_nixl"]["nixl_probe_lock_mode"], "LOCK_SH | LOCK_NB")


if __name__ == "__main__":
    unittest.main()
