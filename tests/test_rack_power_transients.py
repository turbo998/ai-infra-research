from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.rack_power_transients.run import (
    DEFAULT_INPUT,
    DEFAULT_JSON,
    evaluate_transients,
)


class TestRackPowerTransients(unittest.TestCase):
    def test_deterministic_results_match_committed_json(self) -> None:
        config = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        computed = evaluate_transients(config)
        committed = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(computed, committed)

    def test_electro_thermal_invariants(self) -> None:
        config = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        res = evaluate_transients(config)
        s = res["summary"]
        self.assertGreater(s["gb200_unmitigated_cluster_dp_dt_mw_per_s"], 2500.0)
        self.assertGreater(s["gb200_dp_dt_reduction_ratio"], 200.0)
        self.assertAlmostEqual(s["gb200_cbu_usable_energy_kj"], 43.88, places=2)
        self.assertGreater(s["rubin_600kw_cbu_farads_48v"], s["gb200_cbu_farads_48v"] * 20.0)
        self.assertGreater(s["gb200_dummy_burn_energy_overhead_pct"], 5.0)

    def test_cli_check_flag(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "rack_power_transients" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
