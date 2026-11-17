import unittest

from experiments.enterprise_edge_cloud_hybrid_npu_slm_routing_selection.run import build_results


class TestEnterpriseEdgeCloudHybridSelection(unittest.TestCase):
    def test_edge_cloud_cascade_and_cost_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 399)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_task_accuracy_pct"], 95.2)
        self.assertEqual(h["accuracy_gain_t4_vs_t2_pts"], 23.2)
        self.assertEqual(h["ttft_speedup_t4_vs_t1_x"], 5.64)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 78.2)
        self.assertEqual(h["tier4_availability_pct"], 99.98)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["task_accuracy_pct"], tiers[2]["task_accuracy_pct"])
        self.assertLess(tiers[3]["monthly_cloud_gpu_cost_usd"], tiers[2]["monthly_cloud_gpu_cost_usd"])


if __name__ == "__main__":
    unittest.main()
