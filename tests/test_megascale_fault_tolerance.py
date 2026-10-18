from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.megascale_fault_tolerance.run import (
    DEFAULT_INPUT,
    DEFAULT_JSON,
    evaluate_fault_tolerance,
)


class TestMegascaleFaultTolerance(unittest.TestCase):
    def test_deterministic_results_match_committed_json(self) -> None:
        config = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        computed = evaluate_fault_tolerance(config)
        committed = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(computed, committed)

    def test_ettr_and_sdc_invariants(self) -> None:
        config = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        res = evaluate_fault_tolerance(config)
        s = res["summary"]
        self.assertGreater(s["gpus_16k_nvrx_ettr_pct"], 94.0)
        self.assertGreater(s["gpus_64k_nvrx_ettr_pct"], 90.0)
        self.assertLess(s["gpus_64k_slurm_ettr_pct"], 25.0)
        self.assertGreater(s["gpus_64k_nvrx_vs_slurm_gain_pts"], 70.0)
        self.assertEqual(s["sdc_orthogonal_abft_detection_pct"], 100.0)
        self.assertLess(s["sdc_orthogonal_abft_overhead_pct"], 0.1)

    def test_cli_check_flag(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "megascale_fault_tolerance" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
