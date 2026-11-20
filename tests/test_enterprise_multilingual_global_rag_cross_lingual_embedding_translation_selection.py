from __future__ import annotations

import unittest
from experiments.enterprise_multilingual_global_rag_cross_lingual_embedding_translation_selection.run import build_payload


class TestEnterpriseMultilingualGlobalRagCrossLingualEmbeddingTranslationSelection(unittest.TestCase):
    def test_tier4_achieves_highest_cross_lingual_recall_and_sku_precision_at_1x_storage(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreater(tier4["cross_lingual_recall_at_10_pct"], 95.0)
        self.assertGreater(
            tier4["cross_lingual_recall_at_10_pct"],
            tier1["cross_lingual_recall_at_10_pct"] + 50.0,
        )
        self.assertGreater(
            tier4["domain_sku_and_acronym_precision_pct"],
            tier3["domain_sku_and_acronym_precision_pct"] + 30.0,
        )
        self.assertEqual(tier4["index_storage_and_translation_multiplier"], 1.0)
        self.assertLess(
            tier4["query_retrieval_p95_latency_ms"],
            tier3["query_retrieval_p95_latency_ms"] * 0.15,
        )
        self.assertLess(
            tier4["monthly_15m_doc_global_rag_tco_usd"],
            tier2["monthly_15m_doc_global_rag_tco_usd"] * 0.10,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
