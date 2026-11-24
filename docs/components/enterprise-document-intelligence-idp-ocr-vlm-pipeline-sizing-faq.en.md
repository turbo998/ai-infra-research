# Enterprise Document Intelligence (IDP) & Multimodal VLM Pipeline Architecture: `3-Tier Complexity Router vs. Managed Cloud OCR` Selection & Sizing Guide

![Enterprise Document Intelligence 3-Tier Complexity Router Summary](../../assets/svg/enterprise-document-intelligence-idp-ocr-vlm-pipeline-sizing-faq-summary.svg)

> **Target Persona**: Cloud & AI Solution Architect  
> **Core Scenarios**: Financial statement spreading, insurance claims underwriting, contract compliance extraction, and engineering PDF ingestion (`500K–50M pages/month`)  
> **Interactive H5 Learning Workbench**: [`docs/h5/document-intelligence-idp-vlm-pipeline-sizer.html`](../h5/document-intelligence-idp-vlm-pipeline-sizer.html)

---

## 1. Executive Summary

1. **Why Sending Every PDF Page to a `72B` Vision-Language Model Wastes `87%` of GPU Capacity**:
   - Rendering every page at 300 DPI (`~1,820` visual tokens) through `Qwen2.5-VL-72B-FP8` takes `1.85s/page` on `H100` (`$4.20 / 1,000 pages`), even though **`65%` of enterprise PDF pages are born-digital vector text** and **`25%` are standard scanned layouts**.
2. **3-Tier Page Complexity Router (`1.5ms` CPU Inspection Gate)**:
   - **Tier 1 (`65%` Born-Digital PDFs)**: `PyMuPDF (fitz)` + BBox Table Reconstructor on CPU (`2.0ms/page`, **`$0.02 / 1,000 pages`**).
   - **Tier 2 (`25%` Standard Scanned / Multi-Column / Formulas)**: `MinerU / DocLayout-YOLO + PaddleOCR-v4` on `L40S-48GB` (`85ms/page`, **`$0.38 / 1,000 pages`**).
   - **Tier 3 (`10%` Handwritten / Red-Stamped / Complex Merged Tables)**: `OlmOCR-7B / Qwen2.5-VL-72B-FP8` with Guided HTML/JSON Decoding on `H100` (**`$4.20 / 1,000 pages`** on the `10%` slice only).
   - **Blended Pipeline Cost**: **`$0.528 / 1,000 pages`** (`94.7%` cheaper than Azure AI Document Intelligence / AWS Textract Layout at `$10.00 / 1,000 pages`, and `87.4%` cheaper than 100% VLM routing) while achieving **`96.8%` complex table cell F1**.

---

## 2. Core Architect Q&A

### Q1: How does the `1.5ms` CPU gate classify pages without a GPU pass?
It inspects the PDF object tree for embedded font glyph counts (`> 50` chars -> Tier 1), checks HSV color histograms for overlapping red ink stamps or handwriting curvature (-> Tier 3), and escalates any Tier 2 table whose layout confidence falls below `0.82` to Tier 3.

### Q2: Why must financial tables be emitted as `HTML <table rowspan colspan>` rather than Markdown pipe tables?
Markdown pipe tables cannot represent multi-row or multi-column merged headers (`rowspan`/`colspan`), causing **`31%` column-alignment errors** in financial RAG. Emitting structured HTML `<table>` blocks with automatic cross-page header-path injection raises financial numerical QA accuracy from **`68.4%` to `95.2%`**.

---

## 3. Architecture Comparison Matrix (`10M Pages/Month`)

| Dimension | Cloud Managed OCR (`Azure Doc Intelligence / Textract`) | 100% Frontier VLM (`Qwen2.5-VL-72B` on H100) | 100% Layout OCR (`MinerU / Marker`) | **3-Tier Complexity-Routed IDP (`65/25/10`)** |
|---|---|---|---|---|
| **Complex Table Cell F1** | `88.5%` | `96.4%` | `85.0%` | **`96.8%`** |
| **Cost per 1,000 Pages** | `$10.00` | `$4.20` | `$0.38` | **`$0.528` (`-94.7%`)** |
| **Monthly Cost (`10M` Pages)** | `$100,000 / mo` | `$42,000 / mo` (`18× H100`) | `$3,800 / mo` | **`$5,280 / mo` (`2 CPU + 1 L40S + 2 H100`)** |

---

## 4. Production Checklist

- [ ] Deploy the **`1.5ms` CPU Page Complexity Router** ahead of GPU queues.
- [ ] Emit **`HTML <table rowspan colspan>`** with cross-page table stitching and header-unit propagation.
- [ ] Attach **`[page, x1, y1, x2, y2]` bounding-box provenance** to every extracted field for audit verification.
