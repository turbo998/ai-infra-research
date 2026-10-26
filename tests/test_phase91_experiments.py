"""Unit tests for Phase 91 (priority: 100 — 100-Priority Milestone!) offline experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase91Experiments(unittest.TestCase):
    def _run_and_load(self, exp_name: str) -> dict:
        exp_dir = REPO_ROOT / "experiments" / exp_name
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        subprocess.run([sys.executable, str(run_py)], check=True)
        subprocess.run([sys.executable, str(run_py), "--check"], check=True)
        return json.loads(summary_json.read_text(encoding="utf-8"))

    def test_track_a_flashinfer_mm_fp4_kimi_stepfun_ulysses_prims_tilelang_phase91(self) -> None:
        data = self._run_and_load(
            "flashinfer_mm_fp4_prefetch_kimi_ring_tmem_stepfun_fence_ulysses_prims_tilelang_phase91"
        )
        self.assertEqual(data["phase"], 91)
        self.assertEqual(data["priority"], 100)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["cake_mm_fp4_entry_prefetch"]["chain_ep_stages"], 4)
        self.assertEqual(data["kimi_k3_ring_and_tmem_scrub"]["fixed_slots_sequence"], [0, 1, 2, 3])
        self.assertEqual(data["ulysses_direct_prims_mla_tilelang"]["b300_w6_scatter_speedup_x"], 1.108)

    def test_track_b_sglang_vllm_trtllm_phase91(self) -> None:
        data = self._run_and_load(
            "sglang_rust_chat_diffusion_amd_kda_vllm_spec_dp_kv_offload_p2p_trtllm_ttft_phase91"
        )
        self.assertEqual(data["phase"], 91)
        self.assertEqual(data["priority"], 100)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["sglang_rust_diffusion_amd_kda"]["rust_server_dropped_crates_count"], 6)
        self.assertEqual(data["vllm_spec_dp_and_kv_offload_p2p"]["nixl_p2p_ucx_tls_filter"], "^rc_gda")
        self.assertEqual(data["trtllm_kv_scan_and_gen_admission"]["scan_call_reduction_factor_x"], 16)

    def test_track_c_fa4_megatron_deepep_verl_mooncake_nixl_dynamo_phase91(self) -> None:
        data = self._run_and_load(
            "fa4_sm100_mla_res_megatron_seqpack_deepep_verl_cp_mooncake_ssd_nixl_dynamo_phase91"
        )
        self.assertEqual(data["phase"], 91)
        self.assertEqual(data["priority"], 100)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["fa4_mla_residual_and_rocm_splitkv"]["rocm_ck_safe_max_splits"], 8)
        self.assertEqual(data["megatron_deepep_and_verl_distill"]["verl_fixed_misaligned_rows"], 0)
        self.assertEqual(data["mooncake_nixl_and_dynamo"]["dynamo_kv_router_scalar_heap_allocs"], 0)


if __name__ == "__main__":
    unittest.main()
