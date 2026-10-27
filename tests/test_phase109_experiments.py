from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase109Experiments(unittest.TestCase):
    def _run_and_verify(self, exp_dir_name: str) -> dict:
        exp_dir = ROOT / "experiments" / exp_dir_name
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        subprocess.run([sys.executable, str(run_py)], check=True)
        subprocess.run([sys.executable, str(run_py), "--check"], check=True)
        self.assertTrue(summary_json.exists())
        return json.loads(summary_json.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_cudnn_ulysses_gemm_cutlass_tilelang_phase109(self) -> None:
        data = self._run_and_verify(
            "flashinfer_cudnn_plan_ulysses_gemm_host_minimax_phase109"
        )
        self.assertEqual(data["phase"], 109)
        self.assertEqual(data["priority"], 118)
        self.assertEqual(data["invariants"]["cudnn_non_graph_uploads_per_plan"], 1)
        self.assertEqual(data["invariants"]["ulysses_direct_programs_updated"], 24)
        self.assertEqual(data["invariants"]["gemm_host_dispatch_us"], 59.0)
        self.assertAlmostEqual(data["invariants"]["gemm_dispatch_speedup"], 3.3898, places=3)
        self.assertEqual(data["invariants"]["cutlass_sm89_scale_aliasing_errors"], 0)
        self.assertEqual(data["invariants"]["tilelang_keepdim_rank"], 2)

    def test_track_b_sglang_dsv41_vllm_nvfp4_mla_kda_trtllm_helix_phase109(self) -> None:
        data = self._run_and_verify(
            "sglang_dsv41_kv_cap_vllm_nvfp4_mla_kda_trtllm_helix_phase109"
        )
        self.assertEqual(data["phase"], 109)
        self.assertEqual(data["priority"], 118)
        self.assertEqual(data["invariants"]["dsv41_physical_layout_oom_events"], 0)
        self.assertEqual(data["invariants"]["glm53_kda_conv1d_params_after"], 1)
        self.assertEqual(data["invariants"]["heterogeneous_valid_max_tp"], 16)
        self.assertAlmostEqual(
            data["invariants"]["helix_agg_free_gpu_memory_fraction"], 0.6, places=4
        )
        self.assertAlmostEqual(data["invariants"]["gdn_replay_speedup"], 2.5, places=4)

    def test_track_c_te_tma_megatron_torchtitan_mooncake_kfd_nixl_phase109(self) -> None:
        data = self._run_and_verify(
            "te_tma_sync_sm120_megatron_torchtitan_mooncake_kfd_nixl_phase109"
        )
        self.assertEqual(data["phase"], 109)
        self.assertEqual(data["priority"], 118)
        self.assertEqual(data["invariants"]["tensormap_fence_scope_after"], "gpu")
        self.assertEqual(data["invariants"]["cuda_func_set_attribute_errors"], 0)
        self.assertEqual(data["invariants"]["fixed_flex_compiles"], 1)
        self.assertAlmostEqual(
            data["invariants"]["patched_reported_free_vram_gib"], 16.0, places=4
        )
        self.assertTrue(data["invariants"]["dynamo_canonical_digest_match"])
        self.assertEqual(data["invariants"]["nixl_ucx_trace_context_trailer_bytes"], 26)


if __name__ == "__main__":
    unittest.main()
