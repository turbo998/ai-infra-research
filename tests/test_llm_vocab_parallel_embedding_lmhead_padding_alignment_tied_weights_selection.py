import unittest
from experiments.llm_vocab_parallel_embedding_lmhead_padding_alignment_tied_weights_selection.run import (
    evaluate_vocab_parallel_tiers,
)


class TestVocabParallelSelection(unittest.TestCase):
    def test_tiers_and_speedup(self) -> None:
        res = evaluate_vocab_parallel_tiers()
        self.assertEqual(res["raw_vocab_size"], 151665)
        self.assertEqual(res["padded_vocab_size"], 152064)
        self.assertEqual(res["ghost_rows"], 399)
        self.assertAlmostEqual(res["gemm_speedup_x"], 1.63, places=2)
        t1 = res["tiers"][0]
        t4 = res["tiers"][3]
        self.assertEqual(t1["lm_head_gemm_ms"], 4.85)
        self.assertEqual(t4["lm_head_gemm_ms"], 2.98)
        self.assertEqual(t4["ghost_token_crash_pct"], 0.0)
        self.assertEqual(t4["decision"], "Act automatically")


if __name__ == "__main__":
    unittest.main()
