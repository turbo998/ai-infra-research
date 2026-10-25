from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPhase73Experiments(unittest.TestCase):
    def test_track_a_flashinfer_tilelang_sim(self) -> None:
        script = ROOT / "experiments" / "flashinfer_sm103_qsa_walk_mhc4096_sm120_ce_ring_kimi_k3_tilelang_sim" / "run.py"
        summary_path = script.with_name("summary.json")
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], "Phase 73")
        self.assertEqual(data["priority"], 82)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["qsa_sm103_and_sm120"]["rows_128_walk_splits"], 2)
        self.assertEqual(data["sm120_ce_ring_kimi_k3_and_tilelang"]["tilelang_fixed_consumer_barrier_id"], 4)

    def test_track_b_trtllm_vllm_sglang_sim(self) -> None:
        script = ROOT / "experiments" / "trtllm_rubin_vllm_seed_sglang_msa_nvfp4_sim" / "run.py"
        summary_path = script.with_name("summary.json")
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], "Phase 73")
        self.assertEqual(data["priority"], 82)
        self.assertTrue(all(data["invariants"].values()))
        self.assertEqual(data["sglang_msa_nvfp4_and_disagg"]["nvfp4_bytes_per_token_head"], 72)
        self.assertEqual(data["sglang_msa_nvfp4_and_disagg"]["aligned_dp_prefill_tokens"], [512, 512, 512, 512])

    def test_track_c_mooncake_lmcache_megatron_dynamo_sim(self) -> None:
        script = ROOT / "experiments" / "mooncake_neuron_efa_lmcache_mlu_megatron_dynamo_sim" / "run.py"
        summary_path = script.with_name("summary.json")
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(data["phase"], "Phase 73")
        self.assertEqual(data["priority"], 82)
        self.assertTrue(all(data["invariants"].values()))
        self.assertAlmostEqual(data["mooncake_neuron_efa_and_store"]["aggregate_16dev_aligned_gbps"], 231.16)
        self.assertEqual(data["lmcache_torchtitan_megatron_dynamo"]["dynamo_atomic_contention_reduction_x"], 64)


if __name__ == "__main__":
    unittest.main()
