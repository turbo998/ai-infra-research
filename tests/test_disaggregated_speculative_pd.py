import unittest
from experiments.disaggregated_speculative_pd import run as exp


class TestDisaggregatedSpeculativePD(unittest.TestCase):
    def test_results_check_and_invariants(self):
        self.assertEqual(exp.main(["--check"]), 0)
        res = exp.generate_results()
        self.assertEqual(res["evidence_status"], "own-offline")
        sm = res["summary_metrics"]
        self.assertAlmostEqual(sm["seq_32k_scheme_a_warmup_stall_ms"], 58.741, places=3)
        self.assertAlmostEqual(sm["seq_32k_scheme_b_rdma_overhead_pct"], 30.0, places=2)
        self.assertAlmostEqual(sm["seq_32k_scheme_c_rdma_overhead_pct"], 1.25, places=2)
        self.assertLess(sm["seq_32k_scheme_c_first_step_ms"], 15.0)
        self.assertGreater(sm["seq_32k_scheme_c_speedup_vs_ar"], 3.0)
        self.assertGreater(sm["seq_128k_scheme_c_speedup_vs_scheme_a"], 18.0)

    def test_monotonic_stall_growth_in_scheme_a(self):
        res = exp.generate_results()
        stalls = [
            row["scheme_a_kv_only_decode_reprefill"]["decode_warmup_stall_ms"]
            for row in res["sequence_length_sweep"]
        ]
        self.assertEqual(stalls, sorted(stalls))
        self.assertGreater(stalls[-1], stalls[0] * 10.0)


if __name__ == "__main__":
    unittest.main()
