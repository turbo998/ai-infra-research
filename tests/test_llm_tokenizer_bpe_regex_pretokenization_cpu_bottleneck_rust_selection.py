import unittest
from experiments.llm_tokenizer_bpe_regex_pretokenization_cpu_bottleneck_rust_selection.run import evaluate_tokenizer_detokenizer_overhead


class TestTokenizerDetokenizerOverhead(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_tokenizer_detokenizer_overhead()
        self.assertEqual(res["round"], 291)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_single_thread_regex_bpe_plus_on2_full_history_decode"]["prompt_128k_tokenize_ms"],
            tiers["tier4_disaggregated_zmq_rust_chunk_pool_plus_o1_utf8_window_detok"]["prompt_128k_tokenize_ms"] * 50.0,
        )
        self.assertEqual(
            tiers["tier4_disaggregated_zmq_rust_chunk_pool_plus_o1_utf8_window_detok"]["utf8_cjk_fidelity_pct"],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
