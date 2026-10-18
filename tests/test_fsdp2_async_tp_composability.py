"""Unit tests for experiments/fsdp2_async_tp_composability."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.fsdp2_async_tp_composability.run import RESULTS_PATH, build_results


class TestFsdp2AsyncTpComposability(unittest.TestCase):
    def test_deterministic_results_match(self) -> None:
        payload = build_results()
        on_disk = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(payload, on_disk)
        self.assertEqual(payload["measurement_class"], "own-offline")

    def test_fsdp2_fragmentation_and_hbm_savings(self) -> None:
        payload = build_results()
        self.assertLess(payload["summary"]["fsdp2_max_fragmentation_pct"], 1.5)
        self.assertGreater(payload["summary"]["mixed_regime_hbm_saved_gib"], 6.0)
        for row in payload["fsdp1_vs_fsdp2_memory"]:
            self.assertLess(row["fsdp2_fragmentation_pct"], 1.5)
            self.assertGreater(row["fsdp1_reserved_peak_gib"], row["fsdp2_reserved_peak_gib"])
            self.assertGreater(row["repack_speedup_x"], 7.0)

    def test_async_tp_symmetric_memory_overlap(self) -> None:
        payload = build_results()
        for row in payload["async_tp_symmetric_memory"]:
            self.assertGreaterEqual(row["comm_bubble_hidden_pct"], 90.0)
            self.assertGreater(row["end_to_end_layer_speedup_pct"], 20.0)

    def test_cli_check_passes(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "fsdp2_async_tp_composability" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)


if __name__ == "__main__":
    unittest.main()
