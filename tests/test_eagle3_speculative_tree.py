import json
import unittest

from experiments.eagle3_speculative_tree.run import (
    INPUT_PATH,
    RESULTS_JSON_PATH,
    evaluate_eagle3_speculative_tree,
)


class TestEagle3SpeculativeTree(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
        self.results = evaluate_eagle3_speculative_tree(self.config)

    def test_sealed_results_match(self) -> None:
        sealed = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(self.results, sealed)

    def test_low_bs_speedup_and_high_bs_roofline_collapse(self) -> None:
        s = self.results["summary"]
        self.assertGreater(s["code_math_bs1_tree32_speedup"], 2.5)
        self.assertLess(s["code_math_bs32_tree32_collapsed_speedup"], 1.0)
        self.assertLess(s["general_chat_bs16_tree64_collapsed_speedup"], 0.75)

    def test_adaptive_tree_shrinkage_recovers_throughput(self) -> None:
        s = self.results["summary"]
        self.assertEqual(s["adaptive_bs32_optimal_config"], "chain_s5_k1_v6")
        self.assertGreater(s["adaptive_bs32_recovered_speedup"], 2.3)


if __name__ == "__main__":
    unittest.main()
