from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase115Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        exp_dir = ROOT / "experiments" / rel_dir
        script = exp_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("OK:", res.stdout)
        summary_path = exp_dir / "summary.json"
        self.assertTrue(summary_path.is_file())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_cutlass_tilelang_phase115(self) -> None:
        data = self._run_and_load(
            "flashinfer_norm_combine_qsa_nvfp4_megamoe_ep16_mla_tilelang_phase115"
        )
        self.assertEqual(data["phase"], "Phase 115")
        self.assertEqual(data["priority"], 124)
        m = data["metrics"]
        self.assertAlmostEqual(
            m["fused_norm_combine_packed_bf16x2"]["speedup_x"], 1.4118, places=4
        )
        self.assertAlmostEqual(
            m["qsa_nvfp4_sm103_persistent_tcgen05"]["speedup_x"], 1.3821, places=4
        )
        self.assertAlmostEqual(
            m["mxfp8_megamoe_ep16_one_fence"]["speedup_x"], 1.2632, places=4
        )
        self.assertAlmostEqual(
            m["nvfp4_mla_decode_native_e2m1_e4m3"]["speedup_x"], 1.3451, places=4
        )
        self.assertAlmostEqual(
            m["tilelang_sm121_mma_cutedsl_grid_sync"]["barrier_traffic_reduction_x"],
            256.0,
            places=1,
        )

    def test_track_b_sglang_vllm_trtllm_phase115(self) -> None:
        data = self._run_and_load(
            "sglang_fwd_observer_kv_checksum_kvshard_vllm_gemma4_trtllm_phase115"
        )
        self.assertEqual(data["phase"], "Phase 115")
        self.assertEqual(data["priority"], 124)
        m = data["metrics"]
        self.assertAlmostEqual(
            m["sglang_forward_observer_deferred"]["speedup_x"], 1.2807, places=4
        )
        self.assertAlmostEqual(
            m["sglang_hisparse_dsv4_token_backup"]["copy_reduction_x"], 16.0, places=2
        )
        self.assertEqual(
            m["sglang_kvshard_0sm_allgather"]["zero_sm_ce_sms_reserved"], 0
        )
        self.assertAlmostEqual(
            m["vllm_gemma4_skip_padding_fp8_glm53"]["gemma4_router_speedup_x"],
            3.087,
            places=3,
        )
        self.assertAlmostEqual(
            m["trtllm_grpc_image_sm107_mxfp8_kvcm_retention"]["sm107_mxfp8_moe_speedup_x"],
            1.7215,
            places=4,
        )

    def test_track_c_te_megatron_torchtitan_mooncake_phase115(self) -> None:
        data = self._run_and_load(
            "te_fp8bs_swizzle_megatron_grad_win_pad_scale_mooncake_efa_phase115"
        )
        self.assertEqual(data["phase"], "Phase 115")
        self.assertEqual(data["priority"], 124)
        m = data["metrics"]
        self.assertAlmostEqual(
            m["te_fp8bs_multi_swizzle_launch_bounds"]["speedup_x"], 1.4237, places=4
        )
        self.assertEqual(
            m["megatron_mfsdp_multiplicity_padded_scales"]["premature_reduce_scatter_bugs_after"],
            0,
        )
        self.assertAlmostEqual(
            m["torchtitan_rl_vllm_attention_scaling"]["kl_reduction_x"], 482.0, places=1
        )
        self.assertAlmostEqual(
            m["mooncake_worker_rpc_efa_eagain_reuse"]["ping_latency_reduction_x"],
            500.0,
            places=1,
        )
        self.assertAlmostEqual(
            m["lmcache_ptr_dynamo_dra_nixl_dmabuf"]["lmcache_ptr_speedup_x"],
            6.6071,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
