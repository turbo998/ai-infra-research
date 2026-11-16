import unittest
from experiments.llm_code_fim_token_healing_bpe_prefix_boundary_backtracking_selection.run import (
    evaluate_fim_token_healing_tiers,
)


class TestFimTokenHealingSelection(unittest.TestCase):
    def test_tiers_and_accuracy(self) -> None:
        res = evaluate_fim_token_healing_tiers()
        self.assertAlmostEqual(res["unhealed_accuracy_pct"], 57.8, places=1)
        self.assertAlmostEqual(res["healed_accuracy_pct"], 76.2, places=1)
        self.assertAlmostEqual(res["accuracy_gain_pt"], 18.4, places=1)
        self.assertAlmostEqual(res["ttft_speedup_x"], 4.25, places=2)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["mid_identifier_exact_match_pct"], 57.8)
        self.assertEqual(t4["mid_identifier_exact_match_pct"], 76.2)
        self.assertEqual(t4["gpu_prefix_cache_hit_pct"], 91.4)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
