# Enterprise Document Intelligence (IDP), Layout Parsing & VLM Pipeline Evaluator (Round 381)

Deterministic analytical evaluator comparing four enterprise document parsing and table extraction architectures at a scale of `10,000,000` PDF/scanned pages per month:

1. **Tier 1 (`tier1_naive_pypdf_legacy_ocr`)**: Naive Text Extraction & Legacy OCR (`PyPDF / pdfplumber / Tesseract`, `48.5%` complex table TEDS accuracy, `24.0%` financial sum mismatch rate, `$1,200/mo`).
2. **Tier 2 (`tier2_bruteforce_frontier_vlm_all_pages`)**: Brute-Force 100% Frontier VLM Page-by-Page (`92.0%` complex table TEDS accuracy, `3.80 s/page`, `4.2%` financial sum mismatch rate, `$64,000/mo`).
3. **Tier 3 (`tier3_layout_aware_parser_docling_mineru`)**: Dedicated Layout-Aware Document Engine (`Docling / MinerU / Azure Document Intelligence`, `88.4%` complex table TEDS accuracy, `0.42 s/page`, `5.8%` sum mismatch rate, `$4,800/mo`).
4. **Tier 4 (`tier4_hybrid_layout_router_7b_vlm_accounting_check`)**: Complexity-Routed Hybrid IDP (`82%` Fast Layout Parser + `18%` Compact 7B VLM [`Qwen2.5-VL-7B / olmOCR`] + Cross-Page Table Stitching + Deterministic Accounting Sum Verifier, `96.8%` complex table TEDS accuracy, `0.55 s/page` [`6.9x` faster than Tier 2], `0.1%` sum mismatch rate, `$6,200/mo` [`-90.3%` cost vs. Tier 2]).

## Usage

```bash
python experiments/enterprise_document_intelligence_ocr_layout_vlm_pipeline_selection/run.py
python experiments/enterprise_document_intelligence_ocr_layout_vlm_pipeline_selection/run.py --check
```
