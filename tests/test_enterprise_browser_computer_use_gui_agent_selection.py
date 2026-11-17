import unittest

from experiments.enterprise_browser_computer_use_gui_agent_selection.run import build_results


class TestEnterpriseBrowserComputerUseGuiAgentSelection(unittest.TestCase):
    def test_gui_agent_replay_and_axtree_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 391)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_workflow_success_rate_pct"], 98.2)
        self.assertEqual(h["success_gain_t4_vs_t1_pts"], 26.7)
        self.assertEqual(h["speedup_t4_vs_t1_x"], 11.1)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 94.6)
        self.assertEqual(h["tier4_avg_duration_sec"], 4.6)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertGreater(tiers[3]["workflow_success_rate_pct"], tiers[2]["workflow_success_rate_pct"])
        self.assertLess(tiers[3]["avg_workflow_duration_sec"], tiers[2]["avg_workflow_duration_sec"])


if __name__ == "__main__":
    unittest.main()
