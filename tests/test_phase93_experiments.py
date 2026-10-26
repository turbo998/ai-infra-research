"""Unit tests for Phase 93 deterministic Python 3.12 standard-library experiment suites."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase93Experiments(unittest.TestCase):
    def _run_and_load(self, rel_dir: str) -> dict:
        exp_dir = ROOT / "experiments" / rel_dir
        script = exp_dir / "run.py"
        res = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=str(exp_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(res.returncode, 0)
        summary_path = exp_dir / "summary.json"
        self.assertTrue(summary_path.exists())
        return json.loads(summary_path.read_text(encoding="utf-8"))

    def test_track1_flashinfer_sampling_act_qsa_rubin_mla_tilelang(self) -> None:
        data = self._run_and_load("flashinfer_sampling_act_qsa_rubin_mla_tilelang_phase93")
        self.assertEqual(data["phase"], 93)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["sampling_deferred_reduction"]["saved_reduces_normal_path"], 38)
        self.assertEqual(data["small_batch_act_and_mul_split"]["split_blocks"], 64)
        self.assertEqual(data["qsa_redux_and_rubin_shaved_plan"]["idle_sms_eliminated"], 16)
        self.assertEqual(data["kimi_minimax_dots3_tilelang"]["tilelang_legacy_overflow_elements"], 28672)

    def test_track2_sglang_dsv41_vllm_kv_offload_trtllm(self) -> None:
        data = self._run_and_load("sglang_dsv41_vllm_kv_offload_trtllm_phase93")
        self.assertEqual(data["phase"], 93)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["vllm"]["sync_loaded_requests"], ["req_small_1", "req_small_2"])
        self.assertEqual(data["vllm"]["swa_null_chunks_skipped"], 2)
        self.assertEqual(data["trtllm"]["decoder_runner_uses_private_mapping_attr"], "_mapping.is_last_pp_rank()")

    def test_track3_fa4_sparse_mla_megatron_mamba_te_verl_dynamo(self) -> None:
        data = self._run_and_load("fa4_sparse_mla_megatron_mamba_te_verl_dynamo_phase93")
        self.assertEqual(data["phase"], 93)
        self.assertTrue(all(data["invariants"].values()))
        self.assertAlmostEqual(data["fa4_sparse_mla_training"]["dq_dk_error_reduction_97pct_x"], 15.83, places=2)
        self.assertEqual(data["megatron_te_torchtitan_verl"]["te_mxfp8_per_tile_ctas_4096_rows"], 32)
        self.assertAlmostEqual(data["lmcache_and_dynamo"]["dynamo_kv_router_slot_reduction_pct"], 70.8, places=1)


if __name__ == "__main__":
    unittest.main()
