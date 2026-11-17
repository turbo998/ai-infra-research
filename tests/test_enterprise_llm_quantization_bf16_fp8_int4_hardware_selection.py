import unittest

from experiments.enterprise_llm_quantization_bf16_fp8_int4_hardware_selection.run import build_results


class TestEnterpriseLlmQuantizationHardwareSelection(unittest.TestCase):
    def test_quantization_accuracy_and_throughput_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 384)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_relative_accuracy_retention_pct"], 99.7)
        self.assertEqual(h["throughput_gain_t4_vs_t1_x"], 2.24)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 55.4)
        self.assertEqual(h["accuracy_advantage_t4_vs_t2_pts"], 5.5)
        self.assertEqual(h["tier4_per_gpu_throughput_tok_s"], 762)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["per_gpu_throughput_tok_s"], tiers[2]["per_gpu_throughput_tok_s"])
        self.assertGreater(tiers[3]["enterprise_accuracy_score"], tiers[2]["enterprise_accuracy_score"])


if __name__ == "__main__":
    unittest.main()
