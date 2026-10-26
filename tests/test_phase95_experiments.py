"""Unit tests for Phase 95 (priority: 104) deterministic AI Infra experiment suites."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.flashinfer_kimi_router_ag_matmul_kda_minimax_rmsnorm_ssu_bf16fp4_mxfp8_tilelang_phase95.run import (
    build_summary as build_q1,
)
from experiments.lmcache_pygrpc_nixl_gpunetio_megatron_gtp_te_wgrad_titan_dapo_verl_mooncake_dynamo_phase95.run import (
    build_summary as build_q3,
)
from experiments.sglang_ulysses_dsv4_tbo_gfx950_kda_hicache_router_vllm_dspark_fp8_trtllm_mamba_phase95.run import (
    build_summary as build_q2,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase95Experiments(unittest.TestCase):
    def test_q1_flashinfer_and_tilelang_phase95(self) -> None:
        s = build_q1()
        self.assertEqual(s["phase"], 95)
        self.assertEqual(s["priority"], 104)
        r = s["kimi_router_and_ag_matmul"]
        self.assertEqual(r["num_experts"], 896)
        self.assertEqual(r["top_k"], 16)
        self.assertEqual(r["total_measured_cells"], 28)
        self.assertEqual(r["inflight_arrives_with_tail"], 0)
        self.assertTrue(r["tail_preserves_stage_index"])
        self.assertTrue(r["tail_flips_phase_once"])
        t = s["grouped_mxfp8_and_tilelang_sm120"]
        self.assertEqual(t["scale_load_reduction_ratio"], 8.0)
        self.assertEqual(t["cross_stream_overwrite_races_after_fix"], 0)
        res = subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "flashinfer_kimi_router_ag_matmul_kda_minimax_rmsnorm_ssu_bf16fp4_mxfp8_tilelang_phase95"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_q2_sglang_vllm_trtllm_phase95(self) -> None:
        s = build_q2()
        self.assertEqual(s["phase"], 95)
        u = s["sglang_ulysses_dsv4_gfx950_hicache_router"]
        self.assertEqual(u["hidden_k_copy_us"], 42.0)
        self.assertEqual(u["fixed_decode_child_tokens"], 32)
        self.assertEqual(u["fused_kernel_launches"], 1)
        self.assertEqual(u["cold_routed_to_pending_capped"], 4)
        v = s["vllm_pcp_fp8_batch_invariant_kv_trtllm"]
        self.assertEqual(v["supported_oot_kernel_types_count"], 8)
        self.assertEqual(v["usable_prefix_hit_chunks_after_fix"], 4)
        self.assertEqual(v["trtllm_mamba_view_allocations_after"], 96)
        res = subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "sglang_ulysses_dsv4_tbo_gfx950_kda_hicache_router_vllm_dspark_fp8_trtllm_mamba_phase95"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_q3_lmcache_nixl_megatron_te_titan_verl_mooncake_dynamo_phase95(self) -> None:
        s = build_q3()
        self.assertEqual(s["phase"], 95)
        l = s["lmcache_nixl_mooncake_dynamo"]
        self.assertEqual(l["native_dispatcher_python_threads"], 1)
        self.assertEqual(l["lock_hold_reduction_x"], 13.33)
        self.assertEqual(l["per_key_route_lock_unrelated_key_contention"], 0)
        self.assertEqual(l["expanded_grid_size"], 12)
        m = s["megatron_te_torchtitan_verl"]
        self.assertEqual(m["premature_scratch_reuse_hazards"], 0)
        self.assertEqual(m["preserved_small_signal_delta"], 0.004)
        self.assertEqual(m["rollout_wait_speedup_x"], 1.16)
        self.assertEqual(m["hdo_resume_device_mismatch_errors_after_fix"], 0)
        res = subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "experiments"
                    / "lmcache_pygrpc_nixl_gpunetio_megatron_gtp_te_wgrad_titan_dapo_verl_mooncake_dynamo_phase95"
                    / "run.py"
                ),
                "--check",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
