import json
import subprocess
import sys
import unittest
from pathlib import Path


class TestDocumentAIPDFOCRParsingMinerUDoclingOlmOCRSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.exp_dir = (
            self.repo_root
            / "experiments"
            / "document_ai_pdf_ocr_parsing_mineru_docling_olmocr_vlm_layout_selection"
        )
        self.run_script = self.exp_dir / "run.py"
        self.results_path = self.exp_dir / "results.json"

    def test_run_and_check_results(self) -> None:
        subprocess.run([sys.executable, str(self.run_script)], check=True)
        subprocess.run([sys.executable, str(self.run_script), "--check"], check=True)
        data = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(
            data["Benchmark"],
            "document_ai_pdf_ocr_parsing_mineru_docling_olmocr_vlm_layout_selection",
        )
        tiers = {t["Tier_ID"]: t for t in data["Configuration_Matrix"]}
        self.assertGreater(
            tiers["Tier_4_Hybrid_Docling_MinerU_Plus_olmOCR_QwenVL7B_Crop_Router"]["Downstream_RAG_Complex_Doc_Accuracy_Pct"],
            tiers["Tier_1_Legacy_PyMuPDF_Pdfplumber_Rule_Extraction"]["Downstream_RAG_Complex_Doc_Accuracy_Pct"],
        )

    def test_required_artifacts_exist(self) -> None:
        zh = (
            self.repo_root
            / "docs"
            / "components"
            / "document-ai-pdf-ocr-parsing-mineru-docling-olmocr-vlm-layout-selection-faq.md"
        )
        en = (
            self.repo_root
            / "docs"
            / "components"
            / "document-ai-pdf-ocr-parsing-mineru-docling-olmocr-vlm-layout-selection-faq.en.md"
        )
        svg = (
            self.repo_root
            / "assets"
            / "svg"
            / "document-ai-pdf-ocr-parsing-mineru-docling-olmocr-vlm-layout-selection-faq-summary.svg"
        )
        sizer = (
            self.repo_root
            / "docs"
            / "reports"
            / "document-ai-pdf-ocr-parsing-mineru-docling-olmocr-vlm-layout-sizer.html"
        )
        cards = (
            self.repo_root
            / "docs"
            / "reports"
            / "document-ai-pdf-ocr-parsing-mineru-docling-olmocr-vlm-layout-selection-cards.html"
        )
        for p in (zh, en, svg, sizer, cards):
            self.assertTrue(p.exists(), f"Missing artifact: {p}")


if __name__ == "__main__":
    unittest.main()
