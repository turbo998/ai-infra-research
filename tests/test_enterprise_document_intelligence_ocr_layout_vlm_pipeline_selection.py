import unittest

from experiments.enterprise_document_intelligence_ocr_layout_vlm_pipeline_selection.run import build_results


class TestEnterpriseDocumentIntelligencePipelineSelection(unittest.TestCase):
    def test_hybrid_idp_and_accounting_verifier_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 381)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_table_teds_accuracy_pct"], 96.8)
        self.assertEqual(h["accuracy_gain_t4_vs_t1_pts"], 48.3)
        self.assertEqual(h["accuracy_gain_t4_vs_t2_pts"], 4.8)
        self.assertEqual(h["cost_savings_t4_vs_t2_pct"], 90.3)
        self.assertEqual(h["speedup_t4_vs_t2_x"], 6.9)
        self.assertEqual(h["tier4_financial_sum_mismatch_rate_pct"], 0.1)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["complex_table_teds_accuracy_pct"], tiers[1]["complex_table_teds_accuracy_pct"])
        self.assertLess(tiers[3]["financial_sum_mismatch_rate_pct"], tiers[1]["financial_sum_mismatch_rate_pct"])


if __name__ == "__main__":
    unittest.main()
