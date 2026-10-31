from __future__ import annotations
import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class TestPhase140Experiments(unittest.TestCase):
    def test_track_a_phase140(self) -> None:
        exp = ROOT / "experiments" / "flashinfer_primsts_dsa_lpt_vsa_sm90_triton_phase140" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["jit_reduction_pct"], 87.5)
        self.assertEqual(data["metrics"]["lpt_speedup"], 1.52)

    def test_track_b_phase140(self) -> None:
        exp = ROOT / "experiments" / "sglang_dspark_vllm_musespark_trtllm_fp8lora_kimik3_phase140" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["kv_capacity_multiplier"], 3.56)
        self.assertEqual(data["metrics"]["sleep_backup_saving_pct"], 91.67)

    def test_track_c_phase140(self) -> None:
        exp = ROOT / "experiments" / "megatron_mfsdp_pertoken_pytorch_pdl_mooncake_tent_phase140" / "run.py"
        subprocess.run(["python", str(exp), "--check"], check=True)
        data = json.loads(exp.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(data["metrics"]["per_token_grad_bias_pct"], 0.0)
        self.assertEqual(data["metrics"]["row_plan_speedup"], 25.0)

if __name__ == "__main__":
    unittest.main()
