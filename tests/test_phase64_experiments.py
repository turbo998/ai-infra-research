import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase64Experiments(unittest.TestCase):
    def _run_and_load(self, slug: str) -> dict:
        script = REPO_ROOT / "experiments" / slug / "run.py"
        summary_file = REPO_ROOT / "experiments" / slug / "summary.json"
        subprocess.run([sys.executable, str(script)], check=True)
        subprocess.run([sys.executable, str(script), "--check"], check=True)
        return json.loads(summary_file.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_kcp_carry_qsa_nvfp4_mhc_tilelang(self) -> None:
        data = self._run_and_load("flashinfer-kcp-carry-qsa-nvfp4-mhc-tilelang")
        self.assertEqual(data["phase"], 64)
        self.assertEqual(data["priority"], 73)
        kcp = data["kcp_carry_plane"]
        self.assertEqual(kcp["total_matrix_rows"], 144)
        self.assertEqual(kcp["rows_slower_gt_2pct_both_sessions"], 0)
        self.assertAlmostEqual(kcp["carry_plane_min_mib"], 12.0)
        self.assertAlmostEqual(kcp["carry_plane_max_mib"], 80.0)
        self.assertEqual(kcp["p1_carry_plane_bytes"], 0)

        qsa = data["qsa_nvfp4_decode"]
        self.assertEqual(qsa["total_shapes"], 16)
        self.assertAlmostEqual(qsa["peak_speedup"], 1.127)
        self.assertGreater(qsa["mid_batch_16_32_64_geomean_speedup"], 1.08)

        guards = data["megamhc_and_tilelang_guards"]
        self.assertTrue(guards["megamhc_same_grid_accepted"])
        self.assertTrue(guards["megamhc_mismatched_grid_rejected"])
        self.assertTrue(guards["megamhc_sliced_view_rejected"])
        self.assertTrue(guards["tilelang_vectorize_unit_step_accepted"])
        self.assertTrue(guards["tilelang_vectorize_step2_rejected"])
        self.assertEqual(guards["tilelang_signed_int64_atomic_max_fixed"], 3)

    def test_track_b_sglang_marlin_moe_trtllm_kda_swiglu_vllm(self) -> None:
        data = self._run_and_load("sglang-marlin-moe-trtllm-kda-swiglu-vllm")
        self.assertEqual(data["phase"], 64)
        sgl = data["sglang_marlin_and_heartbeat"]
        self.assertEqual(sgl["mapped_buckets"]["-1"], 0)
        self.assertEqual(sgl["mapped_buckets"]["-99"], 0)
        self.assertEqual(sgl["mapped_buckets"]["0"], 1)
        self.assertEqual(sgl["mapped_buckets"]["256"], 0)
        self.assertEqual(sgl["active_blocks_scheduled"], 10)
        self.assertFalse(sgl["heartbeat_buggy_without_pop"]["alive_after_reconnect_single_miss"])
        self.assertTrue(sgl["heartbeat_fixed_with_pop"]["alive_after_reconnect_single_miss"])

        trt = data["trtllm_swiglu_and_kda"]
        self.assertAlmostEqual(trt["saved_hbm_mib_per_layer"], 448.0)
        self.assertAlmostEqual(trt["activation_hbm_reduction_pct"], 80.0)
        self.assertEqual(trt["bf16_roundings_fused"], 1)
        self.assertEqual(trt["kda_legacy_lines_removed"], 5706)

        vllm = data["vllm_mamba_and_spec_rope"]
        self.assertEqual(vllm["mamba_admission_steps_buggy"], 3)
        self.assertEqual(vllm["mamba_admission_steps_fixed"], 1)
        cases = vllm["spec_rope_cases"]
        self.assertEqual(cases["eagle3_llama3_v2"]["effective_draft_max_pos"], 131072)
        self.assertEqual(cases["draft_model_v2"]["effective_engine_max_len"], 2048)
        self.assertEqual(cases["draft_model_v1"]["effective_engine_max_len"], 8192)
        self.assertEqual(cases["eagle3_yarn_v2"]["effective_engine_max_len"], 4096)

    def test_track_c_megatron_mfsdp_mxfp8_te_permute_mooncake_nixl(self) -> None:
        data = self._run_and_load("megatron-mfsdp-mxfp8-te-permute-mooncake-nixl")
        self.assertEqual(data["phase"], 64)
        meg = data["megatron_mfsdp_and_a2a"]
        self.assertAlmostEqual(meg["allgather_bandwidth_reduction_pct"], 50.0)
        self.assertTrue(meg["mtp_mask_replay_matches_forward"])
        self.assertTrue(meg["mtp_mask_without_snapshot_corrupted"])

        te = data["te_permute_and_grouped_tensor"]
        self.assertEqual(te["kept_destinations"], 16403)
        self.assertEqual(te["dropped_destinations_marked_minus1"], 17)
        self.assertEqual(te["oob_writes_prevented"], 17)
        self.assertEqual(te["grouped_tensor_tracked_buffers_count"], 7)

        mnl = data["mooncake_nixl_lmcache"]
        self.assertAlmostEqual(mnl["mooncake_post_send_before_us"], 6.3)
        self.assertAlmostEqual(mnl["mooncake_post_send_after_us"], 1.9)
        self.assertEqual(mnl["nixl_destructive_interference_bytes"], 256)
        self.assertEqual(mnl["nixl_telemetry_version"], 6)
        self.assertGreater(mnl["lmcache_lookup_speedup_x"], 6.0)


if __name__ == "__main__":
    unittest.main()
