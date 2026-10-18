from __future__ import annotations

import unittest

from experiments.nccl_flight_recorder_hang_triage import run as exp


class NcclFlightRecorderHangTriageTest(unittest.TestCase):
    def test_generate_results_and_check(self) -> None:
        data = exp.generate_results()
        self.assertEqual(data["experiment_id"], "nccl_flight_recorder_hang_triage")
        self.assertEqual(data["evidence_status"], "own-offline")
        sm = data["summary_metrics"]
        self.assertLess(sm["max_triage_latency_ms"], 650.0)
        self.assertEqual(sm["patient_zero_localization_accuracy_pct"], 100.0)
        self.assertEqual(sm["false_positive_victims_killed"], 0)
        self.assertGreater(sm["recovery_speedup_x"], 40.0)
        self.assertGreater(sm["dag_triage_30d_goodput_pct"], 99.5)
        self.assertEqual(exp.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
