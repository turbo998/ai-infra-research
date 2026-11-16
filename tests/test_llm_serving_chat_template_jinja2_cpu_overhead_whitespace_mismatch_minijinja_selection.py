import unittest
from experiments.llm_serving_chat_template_jinja2_cpu_overhead_whitespace_mismatch_minijinja_selection.run import (
    evaluate_chat_template_tiers,
)


class TestChatTemplateSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_chat_template_tiers()
        self.assertAlmostEqual(res["python_jinja2_ms"], 16.80, places=2)
        self.assertAlmostEqual(res["rust_minijinja_cached_ms"], 0.30, places=2)
        self.assertAlmostEqual(res["speedup_x"], 56.0, places=1)
        self.assertAlmostEqual(res["accuracy_gain_pt"], 7.2, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["tool_call_accuracy_pct"], 89.2)
        self.assertEqual(t4["tool_call_accuracy_pct"], 96.4)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
