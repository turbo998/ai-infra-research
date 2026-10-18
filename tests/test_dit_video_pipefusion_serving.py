import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiments.dit_video_pipefusion_serving.run import (
    build_results,
    compute_single_step_parallel_schemes,
    simulate_teacache_step_profile,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "dit_video_pipefusion_serving" / "run.py"
RESULTS = ROOT / "experiments" / "dit_video_pipefusion_serving" / "results.json"


class TestDitVideoPipefusionServing(unittest.TestCase):
    def test_cli_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("dit_video_pipefusion_serving: OK", proc.stdout)

    def test_parallel_schemes_reduce_comm_overhead(self) -> None:
        schemes = {r["scheme"]: r for r in compute_single_step_parallel_schemes()}
        tp8 = schemes["tp8_megatron_allreduce"]
        cfg_ulysses = schemes["xdit_cfg2_ulysses4"]
        pipefusion = schemes["xdit_cfg2_ulysses2_pipefusion2_m4"]
        self.assertGreater(tp8["comm_overhead_pct"], 10.0)
        self.assertLess(cfg_ulysses["comm_overhead_pct"], 2.0)
        self.assertLess(pipefusion["comm_overhead_pct"], 1.0)
        self.assertLess(pipefusion["e2e_50steps_s"], tp8["e2e_50steps_s"])

    def test_teacache_forces_boundary_steps_and_monotonic_skipping(self) -> None:
        p08 = simulate_teacache_step_profile(0.08, 50)
        p12 = simulate_teacache_step_profile(0.12, 50)
        p20 = simulate_teacache_step_profile(0.20, 50)
        for prof in (p08, p12, p20):
            self.assertTrue(prof["first_and_last_forced_compute"])
        self.assertLess(p08["skipped_steps_count"], p12["skipped_steps_count"])
        self.assertLess(p12["skipped_steps_count"], p20["skipped_steps_count"])

    def test_stored_summary_invariants(self) -> None:
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(data, build_results())
        summary = data["summary"]
        self.assertGreater(summary["xdit_plus_teacache_t012_speedup_vs_tp8_x"], 4.2)
        self.assertLess(summary["teacache_t012_psnr_drop_db"], 0.4)


if __name__ == "__main__":
    unittest.main()
