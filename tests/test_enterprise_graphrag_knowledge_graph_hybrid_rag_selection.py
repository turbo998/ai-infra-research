from __future__ import annotations

import unittest
from experiments.enterprise_graphrag_knowledge_graph_hybrid_rag_selection.run import build_payload


class TestEnterpriseGraphragKnowledgeGraphHybridRagSelection(unittest.TestCase):
    def test_tier4_highest_single_and_multihop_accuracy_at_fraction_of_indexing_cost(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreater(tier4["single_hop_fact_qa_accuracy_pct"], 96.0)
        self.assertGreater(tier4["multi_hop_and_global_theme_qa_accuracy_pct"], 96.0)
        self.assertGreater(
            tier4["multi_hop_and_global_theme_qa_accuracy_pct"],
            tier2["multi_hop_and_global_theme_qa_accuracy_pct"] + 35.0,
        )
        self.assertLess(
            tier4["corpus_1m_docs_indexing_cost_usd"],
            tier3["corpus_1m_docs_indexing_cost_usd"] * 0.05,
        )
        self.assertLess(
            tier4["monthly_indexing_and_serving_tco_usd"],
            tier3["monthly_indexing_and_serving_tco_usd"] * 0.15,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
