"""Unit tests for Phase 58 deterministic CPU/offline experiment models."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPhase58Experiments(unittest.TestCase):
    def _run_and_load(self, slug: str) -> dict:
        exp_dir = REPO_ROOT / "experiments" / slug
        script = exp_dir / "run.py"
        summary_file = exp_dir / "summary.json"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Wrote", res.stdout)
        return json.loads(summary_file.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_qsa_nvfp4_gdp_fp4_ar_frost_arena(self) -> None:
        slug = "flashinfer-qsa-nvfp4-gdp-onewave-fp4-ar-csp-frost-nvfp4-dsa-mhc-arena"
        data = self._run_and_load(slug)
        self.assertEqual(data["phase"], 58)
        self.assertEqual(data["priority"], 67)
        self.assertTrue(all(data["validation"].values()))

        qsa = data["qsa_nvfp4_sparse_paged_attention"]
        self.assertEqual(qsa["nvfp4_row_bytes"], 144)
        self.assertAlmostEqual(qsa["byte_compression_ratio"], 3.5556, places=4)
        self.assertAlmostEqual(qsa["max_speedup_x"], 1.861, places=3)

        gdp_ar = data["gdp_onewave_and_fp4_allreduce"]
        self.assertAlmostEqual(gdp_ar["gdp_gb300_max_speedup_x"], 1.19, places=2)
        self.assertTrue(gdp_ar["ctrl_indices_verified"])
        self.assertAlmostEqual(gdp_ar["arch_geomeans"]["vr200_sm107a_1024_rows_peak_x"], 1.1172, places=4)

        frost = data["frost_nvfp4_and_memory_arenas"]
        self.assertEqual(frost["num_supported_activations"], 10)
        self.assertTrue(frost["all_midpoints_verified"])
        self.assertAlmostEqual(frost["arena_reduction_pct"], 97.50, places=2)
        self.assertEqual(frost["dsa_checkpoint_retained_mib"]["nope_after_mib"], 0.0)
        self.assertEqual(frost["mhc_train_state_mib"]["after_mib"], 0.0)

    def test_track_b_trtllm_kvcm2_mixed_nvfp4_mxfp8_vllm_sglang(self) -> None:
        slug = "trtllm-kvcm2-mixed-nvfp4-mxfp8-e8m0-vllm-spec-rope-sglang-hb"
        data = self._run_and_load(slug)
        self.assertEqual(data["phase"], 58)
        self.assertEqual(data["priority"], 67)
        self.assertTrue(all(data["validation"].values()))

        kv = data["trtllm_mixed_precision_kv"]
        self.assertEqual(kv["nvfp4_bytes_per_head"], 72)
        self.assertAlmostEqual(kv["seq_profiles"][-1]["memory_savings_pct"], 70.75, places=2)

        mxfp8 = data["mxfp8_e8m0_subnormal_fix"]
        self.assertEqual(mxfp8["subnormal_nan_cases_caught"], 3)
        self.assertTrue(mxfp8["all_new_scales_finite"])

        vs = data["vllm_spec_rope_and_sglang_heartbeat"]
        self.assertTrue(vs["zero_oob_rotary_gathers"])
        self.assertTrue(vs["int64_precision_preserved"])
        self.assertTrue(vs["sglang_heartbeat"]["reset_survives_first_miss"])

    def test_track_c_megatron_ce_mtp_qb_gtp_lmcache_mooncake(self) -> None:
        slug = "megatron-qb-histogram-mtp-early-bwd-gtp-chains-lmcache-mooncake-dfs"
        data = self._run_and_load(slug)
        self.assertEqual(data["phase"], 58)
        self.assertEqual(data["priority"], 67)
        self.assertTrue(all(data["validation"].values()))

        ce_mtp = data["megatron_ce_and_mtp_early_bwd"]
        self.assertAlmostEqual(ce_mtp["inductor_transient_saved_gib"], 4.0, places=3)
        self.assertAlmostEqual(ce_mtp["mtp_peak_reduction_pct"], 73.83, places=2)

        qb_gtp = data["kimi_k3_qb_and_gtp_chains"]
        self.assertTrue(qb_gtp["exact_single_contribution_on_rerun"])
        self.assertEqual(len(qb_gtp["graphed_chain_ids"]), 3)

        lmc_mc = data["lmcache_and_mooncake_safety"]
        self.assertEqual(lmc_mc["leaked_chunks_with_finally"], 0)
        self.assertTrue(lmc_mc["all_cursor_slices_bijective"])
        self.assertEqual(lmc_mc["mooncake_defer_deletion"]["max_soft_pin_ttl_ms"], 604800000)


if __name__ == "__main__":
    unittest.main()
