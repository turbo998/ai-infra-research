import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPhase67Experiments(unittest.TestCase):
    def test_track_a_flashinfer_cake_gdp_cold_chain_frost_ffi(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "flashinfer-cake-gdp-2head-cold-chain-frost-ffi"
        proc = subprocess.run(
            [sys.executable, str(exp_dir / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS]", proc.stdout)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["all_invariants_passed"])

        gdp = summary["cake_gdp_two_head_and_splitv"]
        self.assertEqual(gdp["shuffles_per_substep_before"], 10)
        self.assertEqual(gdp["shuffles_per_substep_after"], 6)
        self.assertAlmostEqual(gdp["shuffle_reduction_pct"], 40.0)
        self.assertTrue(gdp["reduce_scatter_exact_match"])
        self.assertAlmostEqual(gdp["b200_hv80_b256_speedup"], 1.18)
        self.assertAlmostEqual(gdp["gb300_hv80_b256_speedup"], 1.13)

        autotuner = summary["cold_chain_autotuner"]
        self.assertEqual(autotuner["l2_cache_policy"], "cold-chain:8")
        self.assertTrue(autotuner["cold_chain_picks_true_best"])
        self.assertEqual(autotuner["sample_caps_by_chain_ms"], {"0.01": 10, "4.0": 2, "50.0": 1})

        frost = summary["frost_host_ffi_and_tilelang"]
        self.assertTrue(frost["tilelang_non_unit_step_rejected"])
        self.assertTrue(frost["tilelang_signed_ordering_preserved"])

    def test_track_b_sglang_glm53_cp_bcg_trtllm_kda_conv_vllm_gfx942(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942"
        proc = subprocess.run(
            [sys.executable, str(exp_dir / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS]", proc.stdout)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["all_invariants_passed"])

        trtllm = summary["trtllm_kda_conv_and_gvr_v2"]
        self.assertEqual(trtllm["legacy_kda_deleted_lines"], 7214)
        self.assertEqual(trtllm["gvr_qc_before"], 96)
        self.assertEqual(trtllm["gvr_qc_after"], 160)
        self.assertEqual(trtllm["gvr_smem_saved_bytes"], 4096)
        self.assertEqual(trtllm["gvr_det_vpt10_transformed"], [1024, 5])

        sglang = summary["sglang_glm53_cp_bcg_and_megagate"]
        self.assertEqual(sglang["per_rank_logical_tokens"], [193, 193, 192, 192])
        self.assertAlmostEqual(sglang["megagate_mask_reduction_ratio"], 60.0)
        self.assertEqual(sglang["granian_keep_alive_30s_to_ms"], 30000)

        vllm = summary["vllm_gfx942_and_flydsl"]
        self.assertTrue(vllm["gfx942_head_offset_fits_int32"])
        self.assertEqual(vllm["flydsl_deleted_duplicate_gqa6_lines"], 875)
        self.assertTrue(vllm["exceeds_32bit_voffset_without_base_shift"])

    def test_track_c_megatron_mfsdp_mxfp8_planes_lmcache_lookup_nixl_ring(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "megatron-mfsdp-mxfp8-planes-lmcache-lookup-nixl-ring"
        proc = subprocess.run(
            [sys.executable, str(exp_dir / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("[PASS]", proc.stdout)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["all_invariants_passed"])

        mfsdp = summary["mfsdp_mxfp8_planes_and_te"]
        self.assertAlmostEqual(mfsdp["per_phase_allgather_reduction_pct"], 50.0)
        self.assertEqual(mfsdp["te_kept_tokens"], 103)
        self.assertEqual(mfsdp["te_dropped_tokens"], 17)
        self.assertEqual(mfsdp["te_oob_writes_after_fix"], 0)

        titan = summary["torchtitan_gb300_rl_and_flux"]
        self.assertEqual(titan["old_trainer_gpus"], 256)
        self.assertEqual(titan["new_trainer_gpus"], 128)
        self.assertEqual(titan["expert_fsdp_group_reduction_ratio"], 8)
        self.assertEqual(titan["flux_256px_after_packed_schedule"], [1.0, 0.8318, 0.6225, 0.3547])

        kv = summary["lmcache_mooncake_nixl_dynamo"]
        self.assertEqual(kv["lmcache_matched_tokens"], 64)
        self.assertAlmostEqual(kv["mooncake_chain_latency_reduction_pct"], 69.84)
        self.assertEqual(kv["nixl_buffer_header_bytes"], 512)
        self.assertEqual(kv["dynamo_500tok_shared_atomics_after"], 8)


if __name__ == "__main__":
    unittest.main()
