import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.async_rl_staleness_pareto.run import simulate_rl_scheduling_modes


class TestAsyncRlStalenessPareto(unittest.TestCase):
    def test_straggler_bubble_and_pareto_optimum(self) -> None:
        res = simulate_rl_scheduling_modes()
        self.assertGreater(res["trajectory_distribution"]["sync_straggler_bubble_pct"], 70.0)
        self.assertEqual(res["optimal_operating_point"]["optimal_tau_max"], 4)
        self.assertAlmostEqual(res["optimal_operating_point"]["effective_convergence_speedup_x"], 2.91, places=2)

        sweep = {r["tau_max"]: r for r in res["async_staleness_pareto_sweep"]}
        # Unbounded staleness tau_max=8 has higher raw throughput than tau_max=4, but lower effective convergence
        self.assertGreater(sweep[8]["raw_speedup_vs_sync_x"], sweep[4]["raw_speedup_vs_sync_x"])
        self.assertLess(
            sweep[8]["effective_convergence_speedup_x"], sweep[4]["effective_convergence_speedup_x"]
        )
        # Token-aligned TIS outperforms prefix loss masking at tau_max=4
        self.assertGreater(
            sweep[4]["effective_convergence_rate_tis"], sweep[4]["effective_convergence_rate_masked"]
        )

    def test_cli_check_mode(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "async_rl_staleness_pareto" / "run.py"
        proc = subprocess.run([sys.executable, str(script), "--check"], capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        committed = json.loads((script.parent / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(committed["experiment"], "async_rl_staleness_pareto")


if __name__ == "__main__":
    unittest.main()
