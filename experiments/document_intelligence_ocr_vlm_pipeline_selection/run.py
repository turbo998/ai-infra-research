"""Deterministic evaluator for Enterprise Multimodal Document Intelligence (PDF / Financial Tables / Charts OCR & VLM Parsing) Pipeline Selection (Round 48).

Models the Table Structure Recognition accuracy (TEDS %), Chart & Equation extraction accuracy (%),
overall downstream RAG QA accuracy (OmniDocBench %), ingestion throughput (pages/sec/GPU),
numerical hallucination rate on dense financial tables (%), and cost per 1,000,000 pages across four tiers
on a 1,000,000-page enterprise PDF corpus (60% Native Clean/Multi-Col Text, 25% Complex Financial Tables, 15% Charts/Scans/Equations):
  - Tier 1: Legacy Rule-Based PDF Stream + CPU Tesseract OCR (PyMuPDF / pdfplumber)
  - Tier 2: Multi-Stage Layout Specialist Pipeline (MinerU / Docling: DocLayout-YOLO + RapidOCR + StructEqTable)
  - Tier 3: Uniform End-to-End 7B VLM Page-to-Markdown on Every Page (Unanchored Vision-Only VLM OCR)
  - Tier 4: Page-Complexity Adaptive Router (MinerU + Anchor-Grounded 7B VLM FP8) + ColQwen2 Visual Patch Dual-Index
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_document_intelligence_ocr_vlm_pipeline_matrix() -> dict[str, Any]:
    workload = {
        "corpus_profile": "1,000,000 Enterprise PDF Pages (60% Native Multi-Column Text, 25% Complex Merged Financial Tables, 15% Charts/Diagrams/Scanned Equations)",
        "hardware_baseline": "NVIDIA L40S / H100 Document Ingestion Cluster ($2.30/GPU-hr effective amortized rate)",
        "evaluation_benchmark": "OmniDocBench + Financial Table TEDS (Tree-Edit-Distance-Based Similarity) + Downstream RAG Exact-Number QA",
    }

    tiers = [
        {
            "id": "tier1_legacy_pymupdf_tesseract_rule_stream",
            "name": "Tier 1: 传统规则流提取 PyMuPDF / pdfplumber + CPU Tesseract OCR",
            "name_en": "Tier 1: Legacy Rule-Based PDF Stream + CPU Tesseract OCR (PyMuPDF / pdfplumber)",
            "parsing_architecture": "Raw PDF character stream extraction + heuristic line merging + CPU Tesseract on raster images",
            "chart_and_table_handling": "Scrambles 2-column reading order; loses merged table cell hierarchy; ignores visual charts",
            "table_teds_accuracy_pct": 42.5,
            "chart_equation_accuracy_pct": 14.0,
            "overall_rag_qa_accuracy_pct": 54.2,
            "throughput_pages_per_sec_per_gpu": 45.0,
            "numeric_hallucination_rate_pct": 0.2,
            "cost_per_1m_pages_usd": 18.00,
            "recommended_for": "Plain single-column digital-born TXT/PDFs without tables or charts; strictly unfit for financial reports, engineering manuals, or academic PDFs (45.8% RAG failure rate).",
            "status": "fail",
        },
        {
            "id": "tier2_multistage_layout_mineru_docling_pipeline",
            "name": "Tier 2: 多阶段版面检测专用小模型流水线 (MinerU / Docling: YOLO + 表格/公式专模)",
            "name_en": "Tier 2: Multi-Stage Layout Specialist Pipeline (MinerU / Docling: DocLayout-YOLO + Table/Formula Models)",
            "parsing_architecture": "DocLayout-YOLO bounding-box detection -> Reading-Order LayoutReader -> RapidOCR + StructEqTable + UniMERNet",
            "chart_and_table_handling": "High precision on structured tables & LaTeX equations, but blind to complex visual trend charts/flowcharts",
            "table_teds_accuracy_pct": 86.4,
            "chart_equation_accuracy_pct": 68.5,
            "overall_rag_qa_accuracy_pct": 83.5,
            "throughput_pages_per_sec_per_gpu": 18.5,
            "numeric_hallucination_rate_pct": 0.4,
            "cost_per_1m_pages_usd": 125.00,
            "recommended_for": "High-volume text + table + formula corpora without complex charts or messy scans.",
            "status": "pass",
        },
        {
            "id": "tier3_uniform_end_to_end_vlm_page_to_markdown",
            "name": "Tier 3: 全量页面一刀切 7B VLM 端到端转写 (纯视觉 VLM 逐页生成 Markdown)",
            "name_en": "Tier 3: Uniform End-to-End 7B VLM Page-to-Markdown on 100% of Pages (Unanchored Vision VLM)",
            "parsing_architecture": "Renders every PDF page to 1280×1280 image (~1,600 vision tokens) -> Autoregressive 7B VLM generates ~1,000 tokens/page",
            "chart_and_table_handling": "Strong visual chart understanding, but 7.7x slower/costlier and suffers 3.8% digit transposition/hallucination on dense tables",
            "table_teds_accuracy_pct": 91.2,
            "chart_equation_accuracy_pct": 89.5,
            "overall_rag_qa_accuracy_pct": 88.4,
            "throughput_pages_per_sec_per_gpu": 2.4,
            "numeric_hallucination_rate_pct": 3.8,
            "cost_per_1m_pages_usd": 960.00,
            "recommended_for": "Small batches (<10K pages) of heavily scanned or chart-only decks where compute cost is secondary.",
            "status": "warn",
        },
        {
            "id": "tier4_adaptive_page_router_anchored_vlm_colqwen",
            "name": "Tier 4: 页面复杂度自适应路由 (MinerU + 锚点校准 OlmOCR/VLM) + ColQwen2 视觉双路索引",
            "name_en": "Tier 4: Page-Complexity Adaptive Router (MinerU + Anchor-Grounded 7B VLM) + ColQwen2 Visual Patch Index",
            "parsing_architecture": "1ms Page Inspector routes 60% clean pages -> Fast Native/Layout, 25% tables -> MinerU HTML, 15% charts/scans -> OlmOCR Anchored 7B VLM FP8 + ColQwen2 128d INT8 MaxSim",
            "chart_and_table_handling": "Injects raw PDF text coordinates as VLM prompt anchors (0.3% numeric error) + preserves native visual patches for diagram retrieval",
            "table_teds_accuracy_pct": 94.6,
            "chart_equation_accuracy_pct": 93.2,
            "overall_rag_qa_accuracy_pct": 93.8,
            "throughput_pages_per_sec_per_gpu": 24.8,
            "numeric_hallucination_rate_pct": 0.3,
            "cost_per_1m_pages_usd": 92.00,
            "recommended_for": "Production enterprise knowledge bases (financial filings, chip datasheets, medical records) demanding 93.8% QA accuracy at $92/1M pages (-90.4% cost vs uniform VLM).",
            "status": "pass",
        },
    ]

    t1, t2, t3, t4 = tiers
    summary = {
        "tier4_vs_tier1_qa_gain_pts": round(t4["overall_rag_qa_accuracy_pct"] - t1["overall_rag_qa_accuracy_pct"], 1),
        "tier4_vs_tier3_speedup_ratio": round(
            t4["throughput_pages_per_sec_per_gpu"] / t3["throughput_pages_per_sec_per_gpu"], 2
        ),
        "tier4_cost_reduction_vs_tier3_pct": round(
            (t3["cost_per_1m_pages_usd"] - t4["cost_per_1m_pages_usd"]) / t3["cost_per_1m_pages_usd"] * 100.0, 1
        ),
        "tier4_vs_tier3_hallucination_reduction_ratio": round(
            t3["numeric_hallucination_rate_pct"] / t4["numeric_hallucination_rate_pct"], 2
        ),
    }

    return {
        "experiment_id": "document_intelligence_ocr_vlm_pipeline_selection",
        "date": "2026-11-11",
        "round": 48,
        "provenance": {
            "type": "synthetic_analytical_model_calibrated_to_anonymized_published_benchmarks",
            "independently_reproduced": False,
            "physical_gpu_cluster_measurement": "not_measured",
            "calibration_sources": [
                "OmniDocBench (CVPR 2025/2026): Comprehensive Benchmark for Document Parsing across Text, Tables, Formulas & Charts",
                "MinerU & Docling Technical Reports (Multi-Stage Layout Detection, Reading-Order LayoutReader & HTML Table Reconstruction)",
                "allenai/olmOCR (Document Anchoring with PDF Text Coordinates) & ColPali / ColQwen2 Late-Interaction Visual Patch Retrieval",
            ],
        },
        "workload": workload,
        "tiers": tiers,
        "summary": summary,
    }


def main() -> None:
    results = evaluate_document_intelligence_ocr_vlm_pipeline_matrix()
    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
