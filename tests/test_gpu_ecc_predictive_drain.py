"""Unit tests for experiments/gpu_ecc_predictive_drain."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.gpu_ecc_predictive_drain.run import RESULTS_PATH, build_results


class TestGpuEccPredictiveDrain(unittest.TestCase):
    def test_deterministic_results_match(self) -> None:
        payload = build_results()
        on_disk = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload, on_disk)
        self.assertEqual(payload["measurement_class"], "own-offline")

    def test_markov_monotone_hazard_and_abort_reduction(self) -> None:
        payload = build_results()
        probs = [s["prob_fatal_xid_within_72h"] for s in payload["markov_degradation_states"]]
        self.assertEqual(probs, sorted(probs))
        self.assertGreaterEqual(payload["summary"]["unplanned_abort_reduction_pct"], 88.0)
        self.assertGreater(
            payload["summary"]["predictive_drain_ettr_pct"],
            payload["summary"]["reactive_slurm_ettr_pct"],
        )

    def test_cli_check_passes(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "gpu_ecc_predictive_drain" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
