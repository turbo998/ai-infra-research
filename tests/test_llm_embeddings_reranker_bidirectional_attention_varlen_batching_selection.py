import unittest
from experiments.llm_embeddings_reranker_bidirectional_attention_varlen_batching_selection.run import evaluate_embedding_reranker_varlen_serving


class TestEmbeddingRerankerVarlenServing(unittest.TestCase):
    def test_evaluator_invariants(self):
        res = evaluate_embedding_reranker_varlen_serving()
        self.assertEqual(res["round"], 297)
        self.assertIn("independently_reproduced=false", res["provenance"])
        tiers = {t["tier_id"]: t for t in res["tiers"]}
        self.assertGreater(
            tiers["tier1_rectangular_padded_batching_bs64_x_8192"]["p99_batch_latency_ms"],
            tiers["tier4_varlen_cu_seqlens_plus_zero_kv_embed_or_query_prefix_rerank_fp8"]["p99_batch_latency_ms"] * 12.0,
        )
        self.assertEqual(
            tiers["tier4_varlen_cu_seqlens_plus_zero_kv_embed_or_query_prefix_rerank_fp8"]["padding_compute_waste_pct"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
