from __future__ import annotations
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestPhase139Experiments(unittest.TestCase):
    def test_track_a_phase139(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_cudnn_groupedmm_sm107_gqa_tmem_triton_phase139" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["spill_bytes_after"], 0)
        self.assertEqual(data["metrics"]["workspace_reduction_pct"], 75.0)

    def test_track_b_phase139(self) -> None:
        exp = ROOT / "experiments" / "sglang_npu_dsa_vllm_colbert_moelora_trtllm_beam_phase139" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["colbert_batched_speedup"], 2.85)
        self.assertEqual(data["metrics"]["moelora_speedup"], 1.6)

    def test_track_c_phase139(self) -> None:
        exp = ROOT / "experiments" / "megatron_mfsdp_pytorch_xccl2_lmcache_grpc_dynamo_phase139" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["grpc_unix_speedup"], 1.8)
        self.assertEqual(data["metrics"]["gc_deadlock_rate_after"], 0.0)

if __name__ == "__main__":
    unittest.main()
