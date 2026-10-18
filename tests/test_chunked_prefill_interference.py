import json
from pathlib import Path
import subprocess
import sys
import unittest

from experiments.chunked_prefill_interference.run import (
    build_results,
    solve_optimal_prefill_chunk,
    step_latency_breakdown_ms,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "chunked_prefill_interference" / "run.py"
RESULTS = ROOT / "experiments" / "chunked_prefill_interference" / "results.json"


class TestChunkedPrefillInterference(unittest.TestCase):
    def test_cli_check_passes(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--check"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("chunked_prefill_interference: OK", proc.stdout)

    def test_mixed_interference_increases_with_chunk_and_kv(self) -> None:
        pure_dec = step_latency_breakdown_ms(48, 8192, 0, 4096)
        small_chunk = step_latency_breakdown_ms(48, 8192, 512, 4096)
        large_chunk = step_latency_breakdown_ms(48, 8192, 8192, 4096)
        self.assertEqual(pure_dec["mixed_interference_ms"], 0.0)
        self.assertGreater(small_chunk["mixed_interference_ms"], 0.0)
        self.assertGreater(large_chunk["mixed_interference_ms"], small_chunk["mixed_interference_ms"])
        self.assertGreater(large_chunk["step_latency_ms"], 200.0)

    def test_dynamic_solver_respects_25ms_sla_and_tile_alignment(self) -> None:
        solved_low = solve_optimal_prefill_chunk(16, 4096, tpot_sla_ms=25.0)
        solved_mid = solve_optimal_prefill_chunk(48, 8192, tpot_sla_ms=25.0)
        solved_high = solve_optimal_prefill_chunk(96, 16384, tpot_sla_ms=25.0)
        for row in (solved_low, solved_mid, solved_high):
            self.assertEqual(row["optimal_chunk_tokens"] % 64, 0)
            self.assertLessEqual(row["optimal_step_latency_ms"], 25.0)
        self.assertGreater(solved_low["optimal_chunk_tokens"], solved_mid["optimal_chunk_tokens"])
        self.assertGreater(solved_mid["optimal_chunk_tokens"], solved_high["optimal_chunk_tokens"])

    def test_stored_summary_invariants(self) -> None:
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(data, build_results())
        summary = data["summary"]
        self.assertEqual(summary["optimal_chunk_at_48bs_8k_tokens"], 640)
        self.assertGreater(summary["p99_tpot_reduction_vs_static_8192_x"], 10.0)
        self.assertGreater(summary["sla_goodput_gain_vs_static_2048_x"], 4.0)


if __name__ == "__main__":
    unittest.main()
