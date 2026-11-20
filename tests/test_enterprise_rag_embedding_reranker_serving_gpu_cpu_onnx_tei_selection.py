from __future__ import annotations

import unittest
from experiments.enterprise_rag_embedding_reranker_serving_gpu_cpu_onnx_tei_selection.run import build_payload


class TestEnterpriseRagEmbeddingRerankerServingGpuCpuOnnxTeiSelection(unittest.TestCase):
    def test_tier4_achieves_sub40ms_rerank_and_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertLess(tier4["rerank_top50_p99_latency_ms"], 40.0)
        self.assertLess(
            tier4["rerank_top50_p99_latency_ms"],
            tier1["rerank_top50_p99_latency_ms"] * 0.08,
        )
        self.assertGreater(
            tier4["bulk_ingestion_throughput_chunks_per_sec"],
            tier3["bulk_ingestion_throughput_chunks_per_sec"] * 15,
        )
        self.assertGreater(tier4["ndcg_at_10_retrieval_accuracy_pct"], 91.0)
        self.assertLess(
            tier4["monthly_300_qps_and_100m_chunk_tco_usd"],
            tier2["monthly_300_qps_and_100m_chunk_tco_usd"] * 0.15,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
