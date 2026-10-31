from __future__ import annotations
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestPhase138Experiments(unittest.TestCase):
    def test_track_a_phase138(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_d576_mla_sagefp8_r200_triton_phase138" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertGreater(data["metrics"]["mla_schedule_speedup"], 1.45)
        self.assertEqual(data["metrics"]["routing_window_bias"], 0.0)

    def test_track_b_phase138(self) -> None:
        exp = ROOT / "experiments" / "sglang_musespark_fpm_vllm_postload_trtllm_dsv41_phase138" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["post_load_hook_speedup"], 64.0)
        self.assertEqual(data["metrics"]["multi_listener_retention_pct"], 100.0)

    def test_track_c_phase138(self) -> None:
        exp = ROOT / "experiments" / "megatron_greedy_mfsdp_pytorch_ao_lmcache_nixl_phase138" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertGreater(data["metrics"]["greedy_sampling_speedup"], 3.1)
        self.assertEqual(data["metrics"]["bounded_crash_rate"], 0.0)

if __name__ == "__main__":
    unittest.main()
