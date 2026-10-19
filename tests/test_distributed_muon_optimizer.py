from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from experiments.distributed_muon_optimizer.run import (
    generate_summary,
    simulate_distributed_muon_sharding,
    simulate_newton_schulz_spectral_convergence,
)


class DistributedMuonOptimizerTest(unittest.TestCase):
    def test_quintic_newton_schulz_spectral_collapse(self) -> None:
        spec = simulate_newton_schulz_spectral_convergence()
        self.assertAlmostEqual(spec["initial_condition_number_kappa"], 250.0, places=2)
        step5 = spec["step_trajectories"][5]
        self.assertEqual(step5["ns_steps"], 5)
        self.assertLess(step5["condition_number_kappa"], 1.8)
        self.assertGreater(step5["sigma_min"], 0.65)
        self.assertLess(step5["sigma_max"], 1.25)
        self.assertEqual(spec["matrix_2d_verification"]["gram_dimension_chosen"], 8)
        self.assertAlmostEqual(spec["matrix_2d_verification"]["gram_flops_saved_ratio_x"], 2.25, places=2)

    def test_distributed_sharding_saves_memory_and_comm(self) -> None:
        rows = {r["strategy"]: r for r in simulate_distributed_muon_sharding()}
        adamw = rows["ADAMW_FSDP2_BASELINE"]
        naive = rows["MUON_NAIVE_REPLICATED_ALLGATHER"]
        owner = rows["MUON_OWNER_RANK_ALLGATHER"]
        hybrid = rows["MUON_2D_FSDP_OWNER_TP_GRAM"]
        self.assertLess(hybrid["opt_state_memory_per_gpu_gib"], adamw["opt_state_memory_per_gpu_gib"] * 0.35)
        self.assertLess(owner["total_opt_step_ms"], naive["total_opt_step_ms"] / 300.0)
        self.assertLess(hybrid["total_opt_step_ms"], owner["total_opt_step_ms"])
        self.assertLess(hybrid["step_time_overhead_pct"], 1.5)

    def test_cli_check_passes(self) -> None:
        summary = generate_summary()
        self.assertEqual(summary["experiment_id"], "distributed_muon_optimizer")
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "experiments/distributed_muon_optimizer/run.py", "--check"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
