import unittest
from experiments.rag_contextual_retrieval_late_chunking_colbert_reranking_selection.run import (
    simulate_rag_contextual_late_chunking_selection,
)


class TestRagContextualLateChunkingSelection(unittest.TestCase):
    def test_failure_rate_reduction_and_prefix_cache_savings(self) -> None:
        data = simulate_rag_contextual_late_chunking_selection()
        summary = data["summary"]
        self.assertGreater(summary["failure_rate_reduction_ratio"], 6.0)
        self.assertGreater(summary["prefix_cache_token_hit_rate_pct"], 93.0)
        self.assertGreater(summary["indexing_cost_savings_vs_uncached_pct"], 93.0)
        self.assertGreater(summary["ram_savings_vs_full_colbert_pct"], 93.0)

    def test_s5_beats_naive_and_full_colbert(self) -> None:
        data = simulate_rag_contextual_late_chunking_selection()
        by_id = {s["id"]: s for s in data["scenarios"]}
        s1 = by_id["S1_Naive_Chunk_First_Dense"]
        s2 = by_id["S2_Late_Chunking_8K_Encoder"]
        s3 = by_id["S3_Full_ColBERTv2_PLAID_Index"]
        s4 = by_id["S4_Contextual_Hybrid_No_Prefix_Cache"]
        s5 = by_id["S5_Prefix_Cached_Contextual_BGE_M3_Rerank"]

        self.assertGreater(s2["top20_recall_pct"], s1["top20_recall_pct"] + 10.0)
        self.assertGreater(s5["top20_recall_pct"], s3["top20_recall_pct"])
        self.assertLess(s5["vector_index_ram_gb"], s3["vector_index_ram_gb"] * 0.10)
        self.assertLess(s5["indexing_cost_10m_usd"], s4["indexing_cost_10m_usd"] * 0.10)


if __name__ == "__main__":
    unittest.main()
