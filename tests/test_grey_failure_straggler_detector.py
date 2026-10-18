import unittest
from experiments.grey_failure_straggler_detector import run as exp


class TestGreyFailureStragglerDetector(unittest.TestCase):
    def test_results_check_and_classification(self):
        self.assertEqual(exp.main(["--check"]), 0)
        res = exp.generate_results() if hasattr(exp, "generate_results") else exp.simulate_grey_failure_scenarios()
        self.assertEqual(res["evidence_status"], "own-offline")
        for s in res["scenarios"]:
            self.assertTrue(s["classification_correct"], f"Failed on {s['id']}")
        sm = res["summary_metrics"]
        self.assertAlmostEqual(sm["thermal_straggler_mfu_drop_pct"], 32.94, places=2)
        self.assertAlmostEqual(sm["network_fec_straggler_mfu_drop_pct"], 33.22, places=2)
        self.assertAlmostEqual(sm["compound_straggler_mfu_drop_pct"], 63.0, places=2)
        self.assertAlmostEqual(sm["manual_30d_goodput_pct"], 71.35, places=2)
        self.assertAlmostEqual(sm["auto_30d_goodput_pct"], 98.83, places=2)


if __name__ == "__main__":
    unittest.main()
