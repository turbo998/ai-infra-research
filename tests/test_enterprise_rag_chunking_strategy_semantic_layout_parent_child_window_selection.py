from __future__ import annotations

import unittest
from experiments.enterprise_rag_chunking_strategy_semantic_layout_parent_child_window_selection.run import build_payload


class TestEnterpriseRagChunkingStrategySemanticLayoutParentChildWindowSelection(unittest.TestCase):
    def test_tier4_achieves_highest_faithfulness_and_table_recall_at_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreater(tier4["end_to_end_answer_faithfulness_pct"], 96.0)
        self.assertGreater(
            tier4["table_and_clause_integrity_recall_pct"],
            tier1["table_and_clause_integrity_recall_pct"] + 50.0,
        )
        self.assertGreater(
            tier4["table_and_clause_integrity_recall_pct"],
            tier3["table_and_clause_integrity_recall_pct"] + 25.0,
        )
        self.assertLess(
            tier4["indexing_compute_cost_per_1m_pages_usd"],
            tier2["indexing_compute_cost_per_1m_pages_usd"] * 0.05,
        )
        self.assertLess(
            tier4["monthly_20m_page_rag_platform_tco_usd"],
            tier2["monthly_20m_page_rag_platform_tco_usd"] * 0.12,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
