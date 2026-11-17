import unittest

from experiments.enterprise_structured_outputs_json_schema_constrained_decoding_selection.run import build_results


class TestEnterpriseStructuredOutputsConstrainedDecodingSelection(unittest.TestCase):
    def test_structured_output_syntax_and_semantics_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 387)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_json_syntax_valid_pct"], 100.0)
        self.assertEqual(h["tier4_business_semantic_accuracy_pct"], 97.8)
        self.assertEqual(h["semantic_gain_t4_vs_t3_pts"], 13.3)
        self.assertEqual(h["latency_speedup_t4_vs_t1_x"], 5.05)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 47.3)
        self.assertEqual(h["tier4_per_token_overhead_ms"], 0.03)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["business_semantic_accuracy_pct"], tiers[2]["business_semantic_accuracy_pct"])
        self.assertLess(tiers[3]["p99_e2e_latency_ms"], tiers[2]["p99_e2e_latency_ms"])


if __name__ == "__main__":
    unittest.main()
