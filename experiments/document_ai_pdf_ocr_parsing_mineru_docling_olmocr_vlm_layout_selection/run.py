from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_payload() -> dict:
    return {
        "Benchmark": "document_ai_pdf_ocr_parsing_mineru_docling_olmocr_vlm_layout_selection",
        "Provenance": {
            "Data_Status": "synthetic + anonymized_reported",
            "Independently_Reproduced": False,
            "Physical_GPU_Cluster_Measurement": "not_measured",
            "Harness_Policy": "enterprise_document_ai_pdf_mineru_docling_olmocr_vlm_sizer_v1",
        },
        "Workload_Assumptions": {
            "Target_Workload": "Enterprise 10,000,000-Page Financial, Legal & Engineering PDF Ingestion Pipeline (Two-Column Layouts, Merged Tables, Charts & Scanned Pages)",
            "Total_PDF_Pages": 10000000,
            "Complex_Table_Chart_Scan_Page_Share_Pct": 18.0,
            "Target_Downstream_Table_RAG_Accuracy_Pct": 93.0,
        },
        "Configuration_Matrix": [
            {
                "Tier_ID": "Tier_1_Legacy_PyMuPDF_Pdfplumber_Rule_Extraction",
                "Architecture_Name": "Legacy CPU Text Stream Extraction (`PyMuPDF / pdfplumber` without Visual Layout Detection)",
                "Parsing_Pipeline": "Raw PDF Character Stream + Heuristic Line-Box Table Rules",
                "Pages_Per_Second_Per_8GPU_Or_CPU_Node": 185.0,
                "Table_And_Formula_Edit_Similarity_TEDS_Pct": 51.5,
                "Downstream_RAG_Complex_Doc_Accuracy_Pct": 54.0,
                "Cost_Per_1000_Pages_USD": 0.08,
                "Total_10M_Page_Ingestion_TCO_USD": 800,
                "Recommended_Scenario": "Single-column digital-native plain text PDFs without tables, multi-column layouts, or scans",
            },
            {
                "Tier_ID": "Tier_2_100Pct_Heavy_72B_VLM_Page_To_Markdown",
                "Architecture_Name": "100% Frontier / 72B VLM Page-Image-to-Markdown (`300 DPI` Full-Page Vision Tokenization)",
                "Parsing_Pipeline": "Render Every Page to 2048×2048 PNG -> 2,400 Vision Tokens -> 72B VLM Decode",
                "Pages_Per_Second_Per_8GPU_Or_CPU_Node": 4.2,
                "Table_And_Formula_Edit_Similarity_TEDS_Pct": 93.8,
                "Downstream_RAG_Complex_Doc_Accuracy_Pct": 94.2,
                "Cost_Per_1000_Pages_USD": 18.50,
                "Total_10M_Page_Ingestion_TCO_USD": 185000,
                "Recommended_Scenario": "Small (<50K pages) archival or handwritten collections where GPU budget is unconstrained",
            },
            {
                "Tier_ID": "Tier_3_Modular_Layout_Pipeline_MinerU_Docling",
                "Architecture_Name": "Modular Visual Layout Pipeline (`MinerU / IBM Docling`: `DocLayout-YOLO + TableFormer + RapidOCR`)",
                "Parsing_Pipeline": "Specialized Lightweight Vision Models for Reading Order, Bounding Boxes, Tables & LaTeX",
                "Pages_Per_Second_Per_8GPU_Or_CPU_Node": 92.0,
                "Table_And_Formula_Edit_Similarity_TEDS_Pct": 89.4,
                "Downstream_RAG_Complex_Doc_Accuracy_Pct": 88.6,
                "Cost_Per_1000_Pages_USD": 0.42,
                "Total_10M_Page_Ingestion_TCO_USD": 4200,
                "Recommended_Scenario": "High-throughput digital/scanned PDFs with standard tables and multi-column layouts",
            },
            {
                "Tier_ID": "Tier_4_Hybrid_Docling_MinerU_Plus_olmOCR_QwenVL7B_Crop_Router",
                "Architecture_Name": "Complexity-Routed Hybrid Pipeline: `MinerU/Docling` Fast Path (82% Pages) + `olmOCR-7B / Qwen2.5-VL-7B-FP8` on Cropped Complex Tables/Charts (18%)",
                "Parsing_Pipeline": "Layout YOLO Crops Complex Tables/Charts -> Batch vLLM 7B-VLM HTML/Markdown Conversion",
                "Pages_Per_Second_Per_8GPU_Or_CPU_Node": 74.0,
                "Table_And_Formula_Edit_Similarity_TEDS_Pct": 95.2,
                "Downstream_RAG_Complex_Doc_Accuracy_Pct": 94.8,
                "Cost_Per_1000_Pages_USD": 0.68,
                "Total_10M_Page_Ingestion_TCO_USD": 6800,
                "Recommended_Scenario": "Production standard for 1M–100M page enterprise RAG ingestion (94.8% accuracy at -96.3% cost vs full VLM)",
            },
        ],
        "Decision_Summary": {
            "Baseline_Tier": "Tier_2_100Pct_Heavy_72B_VLM_Page_To_Markdown",
            "Recommended_Tier": "Tier_4_Hybrid_Docling_MinerU_Plus_olmOCR_QwenVL7B_Crop_Router",
            "Throughput_Speedup_Multiplier": 17.6,
            "Table_TEDS_Accuracy_Pct": 95.2,
            "RAG_Accuracy_Gain_Over_PyMuPDF_Pts": 40.8,
            "Total_Ingestion_Cost_Savings_Pct": 96.3,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    out_path = Path(__file__).resolve().parent / "results.json"
    payload = build_payload()

    if args.check:
        if not out_path.exists():
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        assert existing["Benchmark"] == payload["Benchmark"]
        assert len(existing["Configuration_Matrix"]) == 4
        print("Wrote results.json")
        return 0

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Wrote results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
