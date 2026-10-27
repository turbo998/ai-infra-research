"""Unit tests for Phase 111 Track A, Track B, and Track C experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestPhase111Experiments(unittest.TestCase):
    def test_track_a_flashinfer_fa4_tilelang_phase111(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "flashinfer_frost_w8a16_fa4_sparse_mla_tilelang_phase111"
        res = subprocess.run(
            [sys.executable, str(exp_dir / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Wrote", res.stdout)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 111)
        self.assertEqual(summary["frost_w8a16_and_ep"]["comm_volume_reduction_ratio"], 8.0)
        self.assertTrue(summary["sm120_and_gb300"]["rtx5090_nvfp4_routed_to_program_1"])
        self.assertEqual(summary["sm120_and_gb300"]["gb300_shift_k4096_permille"], 375)
        self.assertTrue(summary["sm120_and_gb300"]["overflows_dynamic_i32_stride"])
        self.assertEqual(summary["sm120_and_gb300"]["ssu_ctas_per_head"]["96"], 2)
        self.assertEqual(summary["fa4_sparse_mla_and_tilelang"]["saved_gib_64k_default"], 32.0)
        self.assertEqual(summary["fa4_sparse_mla_and_tilelang"]["ds_transient_gib_chunked"], 2.0)

    def test_track_b_sglang_vllm_trtllm_phase111(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "sglang_dflash_mla_vllm_moriio_trtllm_helix_cp_phase111"
        res = subprocess.run(
            [sys.executable, str(exp_dir / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Wrote", res.stdout)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 111)
        self.assertEqual(summary["sglang"]["masked_slot0_writes"], 4)
        self.assertEqual(summary["sglang"]["kernel_launches_saved_per_step"], 8)
        self.assertEqual(summary["vllm"]["rdna_tile_8192"], {"num_warps": 16, "num_stages": 1})
        self.assertEqual(summary["vllm"]["strided_block_len"], 38880)
        self.assertEqual(summary["vllm"]["tail_corruption_bytes_prevented"], 40)
        self.assertEqual(summary["trtllm"]["consensus_admitted"], ["req_1"])
        self.assertEqual(summary["trtllm"]["fused_bf16_roundings"], 1)

    def test_track_c_te_megatron_verl_mooncake_nixl_lmcache_phase111(self) -> None:
        exp_dir = REPO_ROOT / "experiments" / "te_nvfp4_ue5m3_megatron_mfsdp_mooncake_nixl_lmcache_phase111"
        res = subprocess.run(
            [sys.executable, str(exp_dir / "run.py"), "--check"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Wrote", res.stdout)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 111)
        self.assertTrue(summary["te_nvfp4_and_router"]["subtraction_erased_tiny_score"])
        self.assertTrue(summary["megatron_and_verl"]["all_turns_same_dp_rank"])
        self.assertEqual(summary["megatron_and_verl"]["quantized_dbuffer_plane_axes"], [[0, 1], [0, 1]])
        self.assertTrue(summary["mooncake_nixl_lmcache"]["nvlink_unregistered_direct"])
        self.assertFalse(summary["mooncake_nixl_lmcache"]["rdma_unregistered_direct"])
        self.assertEqual(summary["mooncake_nixl_lmcache"]["descriptor_reduction_factor"], 32)


if __name__ == "__main__":
    unittest.main()
