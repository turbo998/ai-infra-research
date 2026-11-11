"""Unit tests for Round 48 Document Intelligence & Multimodal PDF OCR/VLM Pipeline Selection Evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from experiments.document_intelligence_ocr_vlm_pipeline_selection.run import (
    evaluate_document_intelligence_ocr_vlm_pipeline_matrix,
)


class TestDocumentIntelligenceOcrVlmPipelineSelection(unittest.TestCase):
    def test_evaluator_invariants(self) -> None:
        data = evaluate_document_intelligence_ocr_vlm_pipeline_matrix()
        self.assertEqual(data["experiment_id"], "document_intelligence_ocr_vlm_pipeline_selection")
        self.assertFalse(data["provenance"]["independently_reproduced"])
        tiers = {t["id"]: t for t in data["tiers"]}
        self.assertGreater(
            tiers["tier4_adaptive_page_router_anchored_vlm_colqwen"]["overall_rag_qa_accuracy_pct"],
            93.0,
        )
        self.assertGreater(
            tiers["tier4_adaptive_page_router_anchored_vlm_colqwen"]["throughput_pages_per_sec_per_gpu"],
            tiers["tier3_uniform_end_to_end_vlm_page_to_markdown"]["throughput_pages_per_sec_per_gpu"] * 10.0,
        )
        self.assertGreater(
            data["summary"]["tier4_cost_reduction_vs_tier3_pct"],
            90.0,
        )

    def test_sealed_results_json_matches_evaluator(self) -> None:
        results_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "document_intelligence_ocr_vlm_pipeline_selection"
            / "results.json"
        )
        if results_path.exists():
            sealed = json.loads(results_path.read_text(encoding="utf-8"))
            live = evaluate_document_intelligence_ocr_vlm_pipeline_matrix()
            self.assertEqual(sealed["summary"], live["summary"])


if __name__ == "__main__":
    unittest.main()
