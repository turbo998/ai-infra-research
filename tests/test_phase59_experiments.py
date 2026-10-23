"""Unit tests for Phase 59 deterministic offline experiments (`priority: 68`)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_A = REPO_ROOT / "experiments" / "flashinfer-nvfp4-sink-fmha-aligned-mok-mhc-h4096"
EXP_B = REPO_ROOT / "experiments" / "trtllm-gvr-v2-blob-store-sglang-hicache-l3-kvshard-dsa"
EXP_C = REPO_ROOT / "experiments" / "megatron-gdp-prefill-mamba-rows-te-cublasmp-lmcache-mooncake-thp"


class TestPhase59Experiments(unittest.TestCase):
    def test_track_a_summary_and_check(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(EXP_A / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        summary = json.loads((EXP_A / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 59)
        self.assertEqual(summary["priority"], 68)
        self.assertEqual(
            summary["nvfp4_attention_sink_jit"]["disambiguated_unique_names_count"],
            4,
        )
        self.assertAlmostEqual(
            summary["cake_fmha_aligned_hd128"]["gb300_max_speedup_x"],
            10.03,
        )
        self.assertEqual(
            summary["cake_mok_issue_tmem"]["total_libraries_3_archs"],
            144,
        )
        self.assertAlmostEqual(
            summary["cake_deepgemm_dense_mqa"]["gb300_max_speedup_x"],
            1.0785,
        )
        self.assertEqual(
            summary["cake_mega_mhc_h4096"]["total_mega_mhc_programs"],
            15,
        )
        self.assertAlmostEqual(
            summary["cake_causal_conv1d_non_coop_bwd"]["max_speedup_8spin_x"],
            108.8,
        )
        self.assertGreater(
            summary["tilelang_int64_atomics"]["buggy_unsigned_cast_mismatches"],
            0,
        )

    def test_track_b_summary_and_check(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(EXP_B / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        summary = json.loads((EXP_B / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 59)
        self.assertEqual(summary["priority"], 68)
        self.assertEqual(summary["trtllm_gvr_v2_topk"]["register_qc"], 160)
        self.assertEqual(
            summary["trtllm_blob_store_driver"]["invalid_prefixes_rejected"],
            3,
        )
        self.assertTrue(
            summary["sglang_hicache_pd_l3_contract"]["decode_mode_skips_linker_lookup"]
        )
        self.assertFalse(
            summary["sglang_kvshard_dsa"]["supports_mha_one_shot_when_sharded"]
        )
        self.assertTrue(summary["sglang_half_warp_qknorm"]["bit_identical_to_full_warp"])
        self.assertEqual(summary["vllm_pp_tower_and_lora"]["fixed_disk_reloads"], 1)

    def test_track_c_summary_and_check(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(EXP_C / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        summary = json.loads((EXP_C / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 59)
        self.assertEqual(summary["priority"], 68)
        self.assertEqual(
            summary["megatron_gdp_prepare"]["launch_reduction_factor_x"],
            14.0,
        )
        self.assertTrue(
            summary["megatron_mtp_cp_boundary_exchange"]["uses_cp_group_not_world"]
        )
        self.assertEqual(
            summary["megatron_live_row_moe_and_mamba2"]["mamba2_z_buffer_reduction_x"],
            256,
        )
        self.assertEqual(
            summary["te_cublasmp_wgrad_reuse"]["row_parallel_linear_bwd_allgathers_after"],
            0,
        )
        self.assertAlmostEqual(
            summary["torchtitan_dist_moe_mega_swiglu"]["mega_fused_speedup_max_x"],
            1.70,
        )
        self.assertTrue(
            summary["lmcache_tags_and_dax_ranges"]["overlapping_range_returns_http_409"]
        )
        self.assertEqual(
            summary["mooncake_efa_thp_budget"]["max_nic_registration_tib_at_2mb"],
            44.0,
        )


if __name__ == "__main__":
    unittest.main()
