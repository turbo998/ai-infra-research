import unittest
from experiments.llm_serving_liveness_probe_false_200_ok_gpu_worker_watchdog_lease_selection.run import (
    evaluate_liveness_watchdog_tiers,
)


class TestLivenessWatchdogSelection(unittest.TestCase):
    def test_tiers_and_detection(self) -> None:
        res = evaluate_liveness_watchdog_tiers()
        self.assertEqual(res["shallow_hang_detection_pct"], 0.0)
        self.assertAlmostEqual(res["synthetic_probe_false_kill_pct"], 42.0, places=1)
        self.assertEqual(res["shm_watchdog_detection_pct"], 100.0)
        self.assertEqual(res["shm_watchdog_false_kill_pct"], 0.0)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["gpu_hang_detection_rate_pct"], 0.0)
        self.assertEqual(t4["gpu_hang_detection_rate_pct"], 100.0)
        self.assertEqual(t4["probe_latency_p99_ms"], 0.05)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
