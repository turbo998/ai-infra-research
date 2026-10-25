"""Unit tests for Phase 75 (priority: 84) AI Infra research experiments."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase75Experiments(unittest.TestCase):
    def test_track_a_flashinfer_cake_norm_combine_w4a16_moe_minimax_nsplit_frost_fp8_tilelang(self) -> None:
        exp_dir = ROOT / "experiments" / "flashinfer_cake_norm_combine_w4a16_moe_minimax_nsplit_frost_fp8_tilelang"
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        subprocess.run([sys.executable, str(run_py), "--check"], check=True)
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 75)
        self.assertEqual(data["priority"], 84)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["cake_fused_norm_combine"]["max_b300_speedup_x"], 1.107)
        self.assertEqual(data["cake_w4a16_split_moe"]["worst_mid_batch_ratio_x"], 0.923)
        self.assertEqual(data["sm120_varlen_and_cudnn_stream_and_tilelang"]["max_safe_padded_tokens"], 299520)

    def test_track_b_trtllm_qwen3vl_graph_gemma4_vllm_cublas_tail_sglang_msa_nvfp4(self) -> None:
        exp_dir = ROOT / "experiments" / "trtllm_qwen3vl_graph_gemma4_vllm_cublas_tail_sglang_msa_nvfp4"
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        subprocess.run([sys.executable, str(run_py), "--check"], check=True)
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 75)
        self.assertEqual(data["priority"], 84)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["trtllm_mm_encoder_graph_and_gemma4"]["gemma4_budgeting"]["padded_patches_per_frame"], 2304)
        self.assertEqual(data["sglang_msa_nvfp4_and_disagg_checksum"]["msa_v1_nvfp4_kv"]["kv_compression_ratio_vs_bf16"], 3.5556)

    def test_track_c_mooncake_memfabric_rl_affinity_lmcache_packed_mp_megatron_dcpv2_torchtitan(self) -> None:
        exp_dir = ROOT / "experiments" / "mooncake_memfabric_rl_affinity_lmcache_packed_mp_megatron_dcpv2_torchtitan"
        run_py = exp_dir / "run.py"
        summary_json = exp_dir / "summary.json"
        subprocess.run([sys.executable, str(run_py), "--check"], check=True)
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], 75)
        self.assertEqual(data["priority"], 84)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["lmcache_packed_tokens_and_hybrid_groups"]["packed_token_codec"]["memory_reduction_ratio_x"], 9.0)
        self.assertEqual(
            data["megatron_dcpv2_and_torchtitan"]["megatron_dcp_v2"]["valid_group_sizes_by_dp_x_cp"]["24"],
            [1, 2, 4, 8, 24],
        )


if __name__ == "__main__":
    unittest.main()
