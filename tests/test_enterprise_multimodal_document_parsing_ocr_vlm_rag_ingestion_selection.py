from __future__ import annotations

import unittest
from experiments.enterprise_multimodal_document_parsing_ocr_vlm_rag_ingestion_selection.run import build_payload


class TestEnterpriseMultimodalDocumentParsingOcrVlmRagIngestionSelection(unittest.TestCase):
    def test_tier4_highest_table_and_chart_accuracy_at_lowest_tco(self) -> None:
        payload = build_payload()
        tiers = payload["tiers"]
        self.assertEqual(len(tiers), 4)
        tier1 = tiers[0]
        tier2 = tiers[1]
        tier3 = tiers[2]
        tier4 = tiers[3]
        self.assertGreater(
            tier4["complex_table_and_multi_column_rag_accuracy_pct"],
            tier1["complex_table_and_multi_column_rag_accuracy_pct"] + 45.0,
        )
        self.assertGreater(tier4["chart_and_embedded_diagram_qa_recall_pct"], 95.0)
        self.assertLess(tier4["cost_per_1000_pages_usd"], tier2["cost_per_1000_pages_usd"] * 0.30)
        self.assertLess(
            tier4["monthly_15m_page_ingestion_and_rework_tco_usd"],
            tier3["monthly_15m_page_ingestion_and_rework_tco_usd"] * 0.20,
        )

    def test_scenarios_and_perturbations_complete(self) -> None:
        payload = build_payload()
        self.assertEqual(len(payload["scenarios"]), 4)
        self.assertEqual(len(payload["perturbation_checks"]), 2)


if __name__ == "__main__":
    unittest.main()
