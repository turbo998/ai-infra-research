from __future__ import annotations
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestPhase141Experiments(unittest.TestCase):
    def test_track_a_phase141(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_a4w4_kimik3_latentmoe_sm103_sfring_triton_phase141" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["sf_traffic_reduction_pct"], 87.5)
        self.assertEqual(data["metrics"]["fsdp2_walk_reduction_pct"], 96.8)

    def test_track_b_phase141(self) -> None:
        exp = ROOT / "experiments" / "sglang_flat_detok_vllm_kvoffload_trtllm_kvcm2_phase141" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["detok_ipc_speedup"], 2.4)
        self.assertEqual(data["metrics"]["typeadapter_reuse_speedup"], 3.2)

    def test_track_c_phase141(self) -> None:
        exp = ROOT / "experiments" / "megatron_mhc_pytorch_precompile_dynamo_poolrelay_phase141" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["online_autotune_saving_pct"], 100.0)
        self.assertEqual(data["metrics"]["active_active_failover_downtime_ms"], 0.0)

if __name__ == "__main__":
    unittest.main()
