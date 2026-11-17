import unittest

from experiments.enterprise_reasoning_model_thinking_budget_speculative_selection.run import build_results


class TestEnterpriseReasoningModelThinkingBudgetSelection(unittest.TestCase):
    def test_thinking_budget_and_speculative_metrics(self) -> None:
        res = build_results()
        self.assertEqual(res["round"], 385)
        h = res["headline_findings"]
        self.assertEqual(h["tier4_task_accuracy_pct"], 94.6)
        self.assertEqual(h["latency_speedup_t4_vs_t1_x"], 13.24)
        self.assertEqual(h["cost_savings_t4_vs_t1_pct"], 84.4)
        self.assertEqual(h["thinking_token_reduction_pct"], 80.9)
        self.assertEqual(h["tier2_truncated_rate_pct"], 18.4)
        self.assertEqual(h["tier4_truncated_rate_pct"], 0.0)

    def test_four_tiers_ordering(self) -> None:
        res = build_results()
        tiers = res["architecture_tiers"]
        self.assertEqual(len(tiers), 4)
        self.assertLess(tiers[3]["avg_e2e_latency_sec"], tiers[2]["avg_e2e_latency_sec"])
        self.assertEqual(tiers[3]["truncated_mid_thought_rate_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
