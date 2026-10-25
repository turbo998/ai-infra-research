"""Unit tests for Phase 78 (priority: 87) experiments."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase78Experiments(unittest.TestCase):
    def test_track_a_flashinfer_nvfp4_mla_moe_allreduce_softmax(self) -> None:
        exp_dir = ROOT / "experiments" / "flashinfer_nvfp4_mla_decode_moe_allreduce_union_softmax_shared_tile_minimax_h3"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 78)
        self.assertEqual(summary["priority"], 87)
        self.assertAlmostEqual(summary["nvfp4_mla_decode"]["gb300_max_speedup"], 1.1243, places=3)
        self.assertAlmostEqual(summary["moe_allreduce_union"]["b300_max_speedup"], 2.6683, places=3)
        self.assertAlmostEqual(summary["softmax_and_minimax_h3"]["softmax_gb300_max_speedup"], 1.1728, places=3)

    def test_track_b_sglang_push_allreduce_dsa_qsa_vllm_trtllm(self) -> None:
        exp_dir = ROOT / "experiments" / "sglang_push_allreduce_norm_dsa_cp_overlap_qsa_gather_vllm_batch_shard_spec_trtllm"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 78)
        self.assertEqual(summary["priority"], 87)
        self.assertGreater(summary["sglang"]["dsa_cp_max_speedup"], 1.73)
        self.assertGreater(summary["sglang"]["qsa_gather_max_speedup"], 15.0)
        self.assertGreaterEqual(summary["vllm_and_trtllm"]["staged_write_max_wave_speedup"], 16.0)

    def test_track_c_verl_mxfp8_deepgemm_mooncake_xpu_dynamo(self) -> None:
        exp_dir = ROOT / "experiments" / "verl_mxfp8_megatron_sglang_vllm_deepgemm_refit_mooncake_xpu_p2p_dynamo_kvcr"
        subprocess.run([sys.executable, str(exp_dir / "run.py"), "--check"], check=True)
        summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["phase"], 78)
        self.assertEqual(summary["priority"], 87)
        self.assertTrue(summary["verl_logprob_and_veomni"]["probs_diff_blind_to_1ulp"])
        self.assertGreater(summary["verl_mxfp8_and_refit"]["vllm_mxfp4_qat_kl_reduction_ratio"], 5.6)
        self.assertEqual(
            summary["mooncake_and_dynamo"]["dynamo_output_token_batching"]["shared_atomic_reduction_factor"],
            64,
        )


if __name__ == "__main__":
    unittest.main()
