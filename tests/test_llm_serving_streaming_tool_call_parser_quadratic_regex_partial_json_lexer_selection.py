import unittest
from experiments.llm_serving_streaming_tool_call_parser_quadratic_regex_partial_json_lexer_selection.run import (
    evaluate_streaming_tool_parser_tiers,
)


class TestStreamingToolCallParserSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_streaming_tool_parser_tiers()
        self.assertAlmostEqual(res["quadratic_4k_ms"], 42.50, places=2)
        self.assertAlmostEqual(res["incremental_4k_ms"], 0.09, places=2)
        self.assertAlmostEqual(res["speedup_4k_x"], 472.2, places=1)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["argument_stream_corruption_pct"], 16.4)
        self.assertEqual(t4["argument_stream_corruption_pct"], 0.0)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
