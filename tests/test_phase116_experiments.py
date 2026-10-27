from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestPhase116Experiments(unittest.TestCase):
    def _run_suite(self, suite_name: str) -> dict:
        script = REPO_ROOT / "experiments" / suite_name / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("OK:", res.stdout)
        summary_path = REPO_ROOT / "experiments" / suite_name / "summary.json"
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_cutlass_tilelang_phase116(self) -> None:
        data = self._run_suite(
            "flashinfer_cake_fmha_moe_ep_mhc_mla_ssu_vision_cutlass_tilelang_phase116"
        )
        self.assertEqual(data["evidence_mode"], "own-offline")
        self.assertTrue(data["All_30_Sources_Mapped"])
        self.assertTrue(data["cake_fmha_fp8_paged_context"]["static_first_tile_enabled"])
        self.assertEqual(data["cake_fmha_fp8_paged_context"]["round0_atomic_reduction_ratio"], 1.0)
        self.assertGreater(data["megamoe_and_mhc"]["mega_mhc_sm107a_two_warp_fold_speedup"], 1.2)
        self.assertGreater(data["cute_dsl_mla_split_cap_guard"]["mla_split_cap_guard_speedup"], 1.25)
        self.assertEqual(
            data["ssu_vision_cutlass_tilelang"]["tilelang_lower_trace_host_device_overwrite_conflicts"],
            0,
        )

    def test_track_b_sglang_vllm_trtllm_phase116(self) -> None:
        data = self._run_suite(
            "sglang_mtp_graph_qsa_bcg_vllm_numa_dspark_kvcr_spec_trtllm_phase116"
        )
        self.assertEqual(data["evidence_mode"], "own-offline")
        self.assertTrue(data["All_30_Sources_Mapped"])
        self.assertGreater(data["sglang_mtp_qsa_bcg"]["mtp_decode_speedup"], 1.2)
        self.assertEqual(data["sglang_mtp_qsa_bcg"]["qsa_d2d_eliminated_ratio"], 1.0)
        self.assertEqual(data["vllm_numa_dspark_kvcr_spec"]["glm53_draft_kv_saved_gb_per_rank"], 5.38)
        self.assertEqual(
            data["vllm_numa_dspark_kvcr_spec"]["spec_decode_grammar_invalid_rejection_bias"],
            0.0,
        )
        self.assertEqual(
            data["trtllm_gh200_and_comm_matrix"]["trtllm_comm_op_matrix_import_shadow_failures"],
            0,
        )

    def test_track_c_te_megatron_mooncake_dynamo_nixl_phase116(self) -> None:
        data = self._run_suite(
            "te_moe_offset_mxfp8_megatron_metrics_kv_handoff_rope_mooncake_phase116"
        )
        self.assertEqual(data["evidence_mode"], "own-offline")
        self.assertTrue(data["All_30_Sources_Mapped"])
        self.assertTrue(data["te_moe_offset_and_mxfp8"]["int32_overflows"])
        self.assertFalse(data["te_moe_offset_and_mxfp8"]["int64_overflows"])
        self.assertEqual(
            data["megatron_metrics_kv_handoff_and_rope"]["kv_handoff_duplicate_reduction_ratio"],
            0.5,
        )
        self.assertEqual(
            data["megatron_metrics_kv_handoff_and_rope"]["kv_handoff_oob_head_errors"],
            0,
        )
        self.assertEqual(
            data["torchtitan_mooncake_lmcache_dynamo_nixl"][
                "lmcache_undersized_raw_block_silent_corruptions"
            ],
            0,
        )
        self.assertEqual(
            data["torchtitan_mooncake_lmcache_dynamo_nixl"][
                "nixl_mooncake_postxfer_error_leaked_bytes"
            ],
            0,
        )

    def test_phase116_report_and_ledger_integrity(self) -> None:
        report = REPO_ROOT / "docs" / "reports" / "ai-infra-phase116-benchmark-cards.html"
        self.assertTrue(report.is_file())
        html = report.read_text(encoding="utf-8")
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("1 / 7", html)
        self.assertIn("7 / 7", html)


if __name__ == "__main__":
    unittest.main()
