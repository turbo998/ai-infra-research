import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.selective_activation_offload import run as sac_mod


class TestSelectiveActivationOffload(unittest.TestCase):
    def test_operator_knapsack_orders_pointwise_above_gemm(self) -> None:
        table = sac_mod.build_operator_knapsack_table()
        top_categories = [row["category"] for row in table[:4]]
        for cat in top_categories:
            self.assertIn("pointwise", cat)
        self.assertGreater(table[0]["eta_bytes_per_gflop"], table[-1]["eta_bytes_per_gflop"] * 1000)

    def test_sac_plus_pcie_gen5_cuts_86pct_memory_under_5pct_overhead(self) -> None:
        results = sac_mod.generate_results()
        findings = results["headline_findings"]
        self.assertAlmostEqual(findings["full_remat_step_overhead_pct"], 33.33, places=2)
        self.assertGreater(findings["sac_plus_pcie_gen5_memory_reduction_pct"], 86.0)
        self.assertLess(findings["sac_plus_pcie_gen5_step_overhead_pct"], 5.0)

    def test_cli_check_passes(self) -> None:
        script = Path(sac_mod.__file__).resolve()
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        data = json.loads(sac_mod.RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["experiment_id"], "selective_activation_offload")


if __name__ == "__main__":
    unittest.main()
