import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.wafer_sram_gpu_cascaded_serving.run import RESULTS_PATH, build_results


class TestWaferSramGpuCascadedServing(unittest.TestCase):
    def test_build_results_matches_sealed_json(self) -> None:
        computed = build_results()
        sealed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(computed, sealed)

    def test_sram_kv_wall_and_heterogeneous_cascade_economics(self) -> None:
        data = build_results()
        metrics = data["summary_metrics"]
        self.assertEqual(metrics["session_64k_kv_gib"], 10.0)
        self.assertGreater(
            metrics["gb200_8gpu_max_hbm_sessions"],
            metrics["wse3_2x_max_64k_sessions"],
        )
        self.assertEqual(metrics["cascade_speedup_vs_ar_x"], 4.45)
        self.assertEqual(metrics["cascade_cost_reduction_x"], 3.7)

    def test_cli_check_passes(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "wafer_sram_gpu_cascaded_serving" / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
