import unittest
from experiments.llm_reasoning_thinking_budget_forcing_empty_content_truncation_stream_parser_selection.run import (
    evaluate_thinking_budget_tiers,
)


class TestThinkingBudgetSelection(unittest.TestCase):
    def test_tiers_and_savings(self) -> None:
        res = evaluate_thinking_budget_tiers()
        self.assertAlmostEqual(res["naive_empty_answer_pct"], 29.4, places=1)
        self.assertEqual(res["tier4_empty_answer_pct"], 0.0)
        self.assertAlmostEqual(res["accuracy_gain_pt"], 29.0, places=1)
        self.assertAlmostEqual(res["kv_memory_reduction_x"], 3.20, places=2)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["empty_answer_truncation_pct"], 29.4)
        self.assertEqual(t4["empty_answer_truncation_pct"], 0.0)
        self.assertEqual(t4["task_completion_accuracy_pct"], 94.8)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
