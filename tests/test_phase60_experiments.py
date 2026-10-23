"""Unit tests for Phase 60 deterministic offline experiments (`priority: 69`)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_A = REPO_ROOT / "experiments" / "flashinfer-sm120-fp8-mla256-mxfp4-k128-dsa-nope-ulysses-tilelang"
EXP_B = REPO_ROOT / "experiments" / "trtllm-swa-branch-snapshot-minimax-rubin-vstride-sglang-vllm"
EXP_C = REPO_ROOT / "experiments" / "megatron-mamba-res-gtp-cg-mtp-cp-te-mla-mxfp8-lmcache-mooncake"
CARD_REPORT = REPO_ROOT / "docs" / "reports" / "ai-infra-phase60-benchmark-cards.html"


class TestPhase60Experiments(unittest.TestCase):
    def test_track_a_summary_and_check(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(EXP_A / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        summary = json.loads((EXP_A / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["all_invariants_passed"])
        self.assertEqual(summary["sm120_fp8_mla256_and_gqa"]["h_q_per_kv"], 12)
        self.assertEqual(summary["sm120_fp8_mla256_and_gqa"]["sm120_dynamic_smem_kib"], 64)
        self.assertEqual(summary["hopper_mxfp4_megamoe_k128"]["eligible_candidates_before"], 3)
        self.assertEqual(summary["hopper_mxfp4_megamoe_k128"]["eligible_candidates_after"], 8)
        self.assertAlmostEqual(
            summary["hopper_mxfp4_megamoe_k128"]["max_compute_reduction_pct"],
            17.72,
        )
        self.assertAlmostEqual(
            summary["dsa_nope_and_ulysses_and_rope_append"]["latent_dim_reduction_pct"],
            11.11,
        )
        self.assertEqual(
            summary["tilelang_fixes"]["topk_old_overflow_elements"],
            28672,
        )
        self.assertEqual(
            summary["tilelang_fixes"]["topk_new_overflow_elements"],
            0,
        )
        self.assertTrue(summary["tilelang_fixes"]["sm8x_metadata_32x4_bijective"])
        self.assertTrue(summary["tilelang_fixes"]["sm8x_metadata_24x4_rejected"])
        self.assertEqual(summary["tilelang_fixes"]["bf16x2_shift_and_exact"], [1.5, -2.25])

    def test_track_b_summary_and_check(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(EXP_B / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        summary = json.loads((EXP_B / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["all_invariants_passed"])
        self.assertEqual(
            summary["trtllm_swa_branch_snapshot"]["committed_fork_point"],
            1472,
        )
        self.assertAlmostEqual(
            summary["trtllm_swa_branch_snapshot"]["sibling_prefill_saved_pct"],
            71.88,
        )
        self.assertEqual(
            summary["trtllm_minimax_rubin_and_vstride"]["physical_v_stride_bytes"],
            28160,
        )
        self.assertEqual(
            summary["trtllm_minimax_rubin_and_vstride"]["v_token_stride_elements"],
            14080,
        )
        self.assertEqual(
            summary["sglang_and_vllm_fixes"]["ltx2_new_start_frames"],
            [0, 16, 32, 48],
        )
        self.assertAlmostEqual(
            summary["sglang_and_vllm_fixes"]["flux3_swiglu_fp8_hbm_reduction_pct"],
            44.44,
        )
        self.assertEqual(
            summary["sglang_and_vllm_fixes"]["vllm_coop_topk_old_per_block"],
            65536,
        )
        self.assertEqual(
            summary["sglang_and_vllm_fixes"]["vllm_coop_topk_new_per_block"],
            256,
        )

    def test_track_c_summary_and_check(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(EXP_C / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        summary = json.loads((EXP_C / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["all_invariants_passed"])
        self.assertEqual(
            len(summary["megatron_gtp_mtp_and_live_rows"]["gtp_graphed_chain_ids"]),
            3,
        )
        self.assertAlmostEqual(
            summary["megatron_gtp_mtp_and_live_rows"]["moe_live_row_work_saved_pct"],
            69.53,
        )
        self.assertAlmostEqual(
            summary["te_and_torchtitan"]["te_mla_kv_rope_mxfp8_speedup"],
            1.363,
        )
        self.assertEqual(
            summary["te_and_torchtitan"]["te_nvfp4_single_group_fixed_scale_tiles"],
            32,
        )
        self.assertEqual(
            summary["lmcache_mooncake_nixl_dynamo"]["lmcache_fixed_leaked_refs"],
            0,
        )
        self.assertFalse(
            summary["lmcache_mooncake_nixl_dynamo"]["mooncake_tent_fixed_slot_collision"]
        )
        self.assertAlmostEqual(
            summary["lmcache_mooncake_nixl_dynamo"]["nixl_same_socket_speedup"],
            4.4,
        )
        self.assertAlmostEqual(
            summary["lmcache_mooncake_nixl_dynamo"]["dynamo_edge_slot_reduction_pct"],
            70.80,
        )

    def test_card_report_csp_and_7_cards(self) -> None:
        self.assertTrue(CARD_REPORT.exists())
        html = CARD_REPORT.read_text(encoding="utf-8")
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("default-src 'none'; style-src 'unsafe-inline'", html)
        for idx in range(1, 8):
            self.assertIn(f"{idx} / 7", html)


if __name__ == "__main__":
    unittest.main()
