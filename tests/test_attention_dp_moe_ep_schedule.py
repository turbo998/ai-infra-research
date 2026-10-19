import json
import subprocess
import sys
import unittest
from pathlib import Path

from experiments.attention_dp_moe_ep_schedule.run import RESULTS_PATH, build_results


class TestAttentionDpMoeEpSchedule(unittest.TestCase):
    def test_build_results_matches_sealed_json(self) -> None:
        computed = build_results()
        sealed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(computed, sealed)

    def test_dp_attention_kv_capacity_and_tbo_gains(self) -> None:
        data = build_results()
        metrics = data["summary_metrics"]
        self.assertEqual(metrics["kv_capacity_multiplier_x"], 9.34)
        self.assertLess(metrics["fp8_tbo_balanced_step_ms"], metrics["unmasked_max_len_step_ms"])
        self.assertEqual(metrics["throughput_gain_x"], 7.12)
        self.assertEqual(metrics["allocator_reserved_reduction_gib"], 257.6)

    def test_cli_check_passes(self) -> None:
        script = Path(__file__).resolve().parents[1] / "experiments" / "attention_dp_moe_ep_schedule" / "run.py"
        proc = subprocess.run(
            [sys.executable, str(script), "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)


if __name__ == "__main__":
    unittest.main()
