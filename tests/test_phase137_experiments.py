from __future__ import annotations
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestPhase137Experiments(unittest.TestCase):
    def test_track_a_phase137(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_kcp_kda_sm103a_triton_phase137" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertGreater(data["metrics"]["halo_speedup"], 2.5)
        self.assertLess(data["metrics"]["probed_die_imbalance"], 0.05)

    def test_track_b_phase137(self) -> None:
        exp = ROOT / "experiments" / "sglang_pd_vllm_ft_ep_trtllm_mamba2_phase137" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["mamba2_mtp_speedup"], 12.8)
        self.assertGreater(data["metrics"]["orphan_reduction_pct"], 95.0)

    def test_track_c_phase137(self) -> None:
        exp = ROOT / "experiments" / "megatron_kda_pytorch_philox_lmcache_dynamo_phase137" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["fixed_overlap_fraction"], 0.0)
        self.assertGreater(data["metrics"]["reclaimed_pages_pct"], 59.0)

if __name__ == "__main__":
    unittest.main()
