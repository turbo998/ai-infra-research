from __future__ import annotations

import unittest

from experiments.batch_invariant_deterministic_inference import run as exp


class BatchInvariantDeterministicInferenceTest(unittest.TestCase):
    def test_generate_results_and_check(self) -> None:
        data = exp.generate_results()
        self.assertEqual(data["experiment_id"], "batch_invariant_deterministic_inference")
        self.assertEqual(data["evidence_status"], "own-offline")
        sm = data["summary_metrics"]
        self.assertEqual(sm["bs1_dynamic_attn_splits"], 64)
        self.assertEqual(sm["bs33_dynamic_attn_splits"], 1)
        self.assertGreater(sm["bs33_dynamic_logit_ulp_drift"], 0)
        self.assertGreater(sm["bs128_dynamic_1000step_divergence_pct"], 95.0)
        self.assertEqual(sm["batch_invariant_max_ulp_drift"], 0)
        self.assertEqual(sm["batch_invariant_divergence_pct"], 0.0)
        self.assertGreaterEqual(sm["min_throughput_overhead_pct"], 4.0)
        self.assertLessEqual(sm["max_throughput_overhead_pct"], 9.0)
        self.assertEqual(exp.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
