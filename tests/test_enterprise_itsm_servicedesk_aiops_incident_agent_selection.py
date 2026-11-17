import unittest

from experiments.enterprise_itsm_servicedesk_aiops_incident_agent_selection.run import build_results


class TestEnterpriseItsmAiopsIncidentAgentSelection(unittest.TestCase):
    def test_itsm_aiops_mttr_and_safety_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 397)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_ticket_automation_rate_pct"], 76.5)
        self.assertEqual(h["automation_gain_t4_vs_t1_pts"], 60.5)
        self.assertEqual(h["mttr_speedup_t4_vs_t1_x"], 16.6)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 72.2)
        self.assertEqual(h["tier3_unsafe_mutation_rate_pct"], 3.8)
        self.assertEqual(h["tier4_unsafe_mutation_rate_pct"], 0.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["avg_mttr_minutes"], tiers[2]["avg_mttr_minutes"])
        self.assertEqual(tiers[3]["unsafe_prod_mutation_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
