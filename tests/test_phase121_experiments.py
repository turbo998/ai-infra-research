"""Unit tests for Phase 121 Track A, Track B, and Track C experiments."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase121Experiments(unittest.TestCase):
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

    def test_track_a_flashinfer_cake_softmax_mm_fp4_vc_qk8_minimax_sampling_cutlass_tilelang(
        self,
    ) -> None:
        s = self._run_and_load(
            "flashinfer_cake_softmax_mm_fp4_vc_qk8_minimax_h3_sampling_cutlass_tilelang_phase121"
        )
        self.assertEqual(s["phase"], 121)
        self.assertEqual(
            s["cake_softmax_fence"]["fence_ptx"], "fence.proxy.async.shared::cta;"
        )
        self.assertEqual(s["cake_softmax_fence"]["stress_errors_after_fix"], 0)
        self.assertEqual(s["cake_mm_fp4_and_qsa"]["mm_fp4_fallback_count"], 3)
        self.assertAlmostEqual(
            s["vc_qk8_and_minimax_h3"]["wan22_p1_vc_qk8_repair_speedup"], 1.4435, places=4
        )
        self.assertEqual(s["vc_qk8_and_minimax_h3"]["minimax_h3_sm103_min_tile_pairs"], 608)
        self.assertAlmostEqual(
            s["autotuner_and_sampling"]["max_sampling_speedup"], 3.7569, places=4
        )
        self.assertTrue(s["autotuner_and_sampling"]["threads_1024_overflows_sm120"])
        self.assertTrue(s["autotuner_and_sampling"]["threads_512_fits_sm120"])
        self.assertTrue(s["cutlass_and_tilelang"]["batch_indexing_restored"])

    def test_track_b_sglang_triton39_ascend_moe_rust_ming_vllm_zipf_trtllm(self) -> None:
        s = self._run_and_load(
            "sglang_triton39_ascend_moe_rust_sidecar_ming_vllm_zipf_trtllm_phase121"
        )
        self.assertEqual(s["phase"], 121)
        self.assertEqual(s["sglang_triton39_and_ascend"]["step3_valid_rows"], 4)
        self.assertEqual(s["sglang_triton39_and_ascend"]["step3_masked_rows"], 28)
        self.assertEqual(s["sglang_rust_and_ming"]["new_sidecars_per_node"], 1)
        self.assertEqual(s["sglang_rust_and_ming"]["ming_hidden_size"], 3840)
        self.assertAlmostEqual(
            s["vllm_frontend_and_zipf"]["zipf_top8_mass_share"], 0.7076, places=4
        )
        self.assertAlmostEqual(
            s["vllm_frontend_and_zipf"]["zipf_max_to_mean_expert_ratio"], 20.1027, places=4
        )
        self.assertEqual(
            s["trtllm_inductor_and_minimax"]["inductor_quiesce_default_value"], "0"
        )
        self.assertTrue(
            s["trtllm_inductor_and_minimax"]["minimax_dit_supports_hybrid_ulysses_tp"]
        )

    def test_track_c_megatron_mfsdp_nccl_te_fa4_mooncake_lmcache_dynamo_nixl(self) -> None:
        s = self._run_and_load(
            "megatron_mfsdp_unpadded_hyperconnection_nccl_gin_te_compile_fa4_mooncake_lmcache_dynamo_nixl_phase121"
        )
        self.assertEqual(s["phase"], 121)
        self.assertEqual(
            s["megatron_mfsdp_and_hyperconnection"]["unpadded_total_numel"],
            s["megatron_mfsdp_and_hyperconnection"]["total_true_numel"],
        )
        self.assertGreater(
            s["megatron_mfsdp_and_hyperconnection"]["unpadded_saved_numel"], 0
        )
        self.assertTrue(s["nccl_te_and_fa4"]["fa4_empty_tile_buggy_is_nan"])
        self.assertEqual(s["nccl_te_and_fa4"]["fa4_empty_tile_fixed_val"], 0.0)
        self.assertTrue(s["mooncake_lmcache_dynamo_nixl"]["mooncake_export_is_page_aligned"])
        self.assertFalse(s["mooncake_lmcache_dynamo_nixl"]["dynamo_gpt_oss_reasoning_disabled"])
        self.assertEqual(
            s["mooncake_lmcache_dynamo_nixl"]["nixl_fixed_leaked_bytes_per_mismatch"], 0
        )


if __name__ == "__main__":
    unittest.main()
