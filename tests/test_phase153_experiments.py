# -*- coding: utf-8 -*-
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestPhase153Experiments(unittest.TestCase):
    def test_track1_flashinfer_triton_tilelang_phase153(self):
        script = ROOT / "experiments" / "flashinfer_kda_tmem_kcp_recompute_triton_tilelang_phase153" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track2_sglang_vllm_trtllm_phase153(self):
        script = ROOT / "experiments" / "sglang_dsv41_eplb_mamba_vllm_mi355x_trtllm_phase153" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

    def test_track3_megatron_pytorch_mooncake_dynamo_nixl_phase153(self):
        script = ROOT / "experiments" / "megatron_mimo_cp_mtp_mooncake_repl_nixl_proxy_phase153" / "run.py"
        res = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)

if __name__ == "__main__":
    unittest.main()
