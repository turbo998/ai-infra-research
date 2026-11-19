# Enterprise Multimodal Document Parsing, OCR & RAG Ingestion Platform Selection (`R471`)

Deterministic evaluation suite comparing four enterprise document parsing and multimodal RAG ingestion architectures across `15,000,000` PDF pages/month (`55%` simple born-digital text, `35%` complex multi-column/tables/scans, `10%` charts & architecture diagrams):

1. **Tier 1**: Basic CPU Rule-Based PDF Extraction (`PyMuPDF / pypdf`) (`48.5%` complex table RAG accuracy, `12.0%` chart recall, `$0.15/1K pages`, `$148,000/mo` TCO including downstream RAG failure rework)
2. **Tier 2**: 100% Cloud-Managed Document AI API (`Azure Document Intelligence Layout / AWS Textract`) (`91.2%` table accuracy, `54.0%` chart recall, `$10.00/1K pages`, `$156,000/mo` TCO)
3. **Tier 3**: 100% Frontier VLM Page-by-Page Rendering (`GPT-4o / Claude Vision` on all pages) (`93.8%` table accuracy, `94.5%` chart recall, `$18.50/1K pages`, `$278,000/mo` TCO)
4. **Tier 4**: Complexity-Routed Hybrid Ingestion (`PyMuPDF` Fast-Path + GPU `Docling / MinerU` + Cropped-Figure VLM) (`96.4%` table/layout accuracy, `95.2%` chart recall, `310 pages/s` throughput, `$2.40/1K pages`, `$36,000/mo` TCO)

## Usage

```bash
python experiments/enterprise_multimodal_document_parsing_ocr_vlm_rag_ingestion_selection/run.py
python experiments/enterprise_multimodal_document_parsing_ocr_vlm_rag_ingestion_selection/run.py --check
python -m unittest tests/test_enterprise_multimodal_document_parsing_ocr_vlm_rag_ingestion_selection.py
```
