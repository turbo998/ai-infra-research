import unittest

from experiments.enterprise_multi_lora_serving_hundred_tenant_adapter_selection.run import build_results


class TestEnterpriseMultiLoraServingSelection(unittest.TestCase):
    def test_multi_lora_consolidation_and_ttft_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 393)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_gpus_required"], 8)
        self.assertEqual(h["gpu_reduction_t4_vs_t1_x"], 15.0)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 93.3)
        self.assertEqual(h["ttft_speedup_t4_vs_t3_x"], 3.16)
        self.assertEqual(h["tier4_p99_ttft_ms"], 215.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["p99_ttft_ms"], tiers[2]["p99_ttft_ms"])
        self.assertGreater(tiers[3]["avg_gpu_utilization_pct"], tiers[2]["avg_gpu_utilization_pct"])


if __name__ == "__main__":
    unittest.main()
