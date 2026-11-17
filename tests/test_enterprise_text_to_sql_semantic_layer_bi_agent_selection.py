import unittest

from experiments.enterprise_text_to_sql_semantic_layer_bi_agent_selection.run import build_results


class TestEnterpriseTextToSqlSemanticLayerSelection(unittest.TestCase):
    def test_semantic_layer_and_ast_gate_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 378)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_business_accuracy_pct"], 96.4)
        self.assertEqual(h["accuracy_gain_t4_vs_t1_pts"], 44.4)
        self.assertEqual(h["accuracy_gain_t4_vs_t2_pts"], 22.9)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 78.7)
        self.assertEqual(h["tier4_runaway_scan_incident_rate_pct"], 0.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["business_metric_accuracy_pct"], tiers[1]["business_metric_accuracy_pct"])
        self.assertEqual(tiers[3]["runaway_scan_incident_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
