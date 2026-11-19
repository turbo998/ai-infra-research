from __future__ import annotations

import unittest
from experiments.enterprise_rag_vector_search_azure_milvus_qdrant_opensearch_pgvector_selection.run import build_payload


class TestEnterpriseRagVectorSearchAzureMilvusQdrantOpensearchPgvectorSelection(unittest.TestCase):
    def test_tiers_filtered_recall_latency_and_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreaterEqual(tier3["filtered_rbac_recall_at_10_pct"], 96.0)
        self.assertGreaterEqual(tier4["filtered_rbac_recall_at_10_pct"], 98.0)
        self.assertLessEqual(tier4["hybrid_Search_p99_latency_ms"], 20.0)
        self.assertGreaterEqual(tier4["exact_part_number_and_acronym_hit_pct"], 97.0)
        self.assertLess(tier4["monthly_search_cluster_and_ops_tco_usd"], tier1["monthly_search_cluster_and_ops_tco_usd"] * 0.35)

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
