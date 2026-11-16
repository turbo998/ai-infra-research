import unittest
from experiments.llm_serving_prompt_logprobs_chunked_prefill_vocab_oom_streaming_lse_selection.run import evaluate_prompt_logprobs_streaming_lse


class TestPromptLogprobsStreamingLse(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_prompt_logprobs_streaming_lse()
        self.assertEqual(res["round"], 316)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertLess(
            tiers["tier4_micro_tiled_streaming_lmhead_lse_plus_cross_chunk_shift_buffer"]["peak_prompt_logits_hbm_gb"],
            0.5,
        )
        self.assertFalse(
            tiers["tier4_micro_tiled_streaming_lmhead_lse_plus_cross_chunk_shift_buffer"]["chunk_boundary_logprob_drop"]
        )


if __name__ == "__main__":
    unittest.main()
