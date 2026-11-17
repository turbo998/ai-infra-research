import unittest

from experiments.enterprise_gpu_sharing_mig_mps_timeslicing_selection.run import build_results


class TestEnterpriseGpuSharingMigMpsSelection(unittest.TestCase):
    def test_gpu_sharing_packing_and_isolation_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 389)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_gpus_required"], 3)
        self.assertEqual(h["gpu_reduction_t4_vs_t1_x"], 9.3)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 89.3)
        self.assertEqual(h["p99_speedup_t4_vs_t2_x"], 4.67)
        self.assertEqual(h["tier2_oom_blast_rate_pct"], 14.5)
        self.assertEqual(h["tier4_oom_blast_rate_pct"], 0.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["p99_inference_latency_ms"], tiers[1]["p99_inference_latency_ms"])
        self.assertGreater(tiers[3]["avg_sm_utilization_pct"], tiers[2]["avg_sm_utilization_pct"])


if __name__ == "__main__":
    unittest.main()
