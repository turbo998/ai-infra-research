from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.fp8_nvfp4_numerics.run import DEFAULT_INPUT, DEFAULT_JSON, evaluate_numerics


class TestFP8NVFP4Numerics(unittest.TestCase):
    def test_deterministic_results_match_committed_json(self) -> None:
        config = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        computed = evaluate_numerics(config)
        committed = json.loads(DEFAULT_JSON.read_text(encoding="utf-8"))
        self.assertEqual(computed, committed)

    def test_physical_and_numerical_invariants(self) -> None:
        config = json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))
        res = evaluate_numerics(config)
        s = res["summary"]
        self.assertGreater(
            s["outlier_100x_fp8_1x128_non_outlier_sqnr_db"],
            s["outlier_100x_fp8_delayed_non_outlier_sqnr_db"] + 9.0,
        )
        self.assertGreater(s["outlier_100x_nvfp4_rht_sqnr_gain_db"], 15.0)
        self.assertLess(s["wgrad_sr_relative_bias_pct"], 2.0)
        self.assertGreater(s["wgrad_rtn_relative_bias_pct"], 30.0)
        self.assertLess(s["deepgemm_128_promo_rel_err_pct"], s["wgmma_no_promo_rel_err_pct"])

    def test_cli_check_flag(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "fp8_nvfp4_numerics" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
