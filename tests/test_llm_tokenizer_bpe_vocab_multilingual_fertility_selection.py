import unittest
from experiments.llm_tokenizer_bpe_vocab_multilingual_fertility_selection.run import (
    evaluate_tokenizer_vocab_tiers,
)


class TestTokenizerVocabSelection(unittest.TestCase):
    def test_tiers_and_fertility(self) -> None:
        data = evaluate_tokenizer_vocab_tiers()
        self.assertFalse(data["independently_reproduced"])
        self.assertEqual(data["physical_gpu_cluster_measurement"], "not_measured")
        tiers = {t["tier_id"]: t for t in data["tiers"]}
        t1 = tiers["tier1_legacy_32k_bpe_utf8_byte_fallback"]
        t4 = tiers["tier4_sweetspot_128k_152k_multilingual_rust_tiktoken"]
        self.assertGreater(t1["tokens_for_100k_cjk_chars"], 140000)
        self.assertLess(t4["tokens_for_100k_cjk_chars"], 61000)
        self.assertGreaterEqual(t4["relative_decode_speedup_cjk"], 2.4)
        self.assertLessEqual(t4["cpu_128k_tokenize_ms"], 5.0)


if __name__ == "__main__":
    unittest.main()
