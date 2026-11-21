# Enterprise Document AI & Complex PDF Parsing Selection Experiment (`MinerU / Docling` vs. `olmOCR / Qwen2.5-VL` vs. `PyMuPDF`)

Reproducible throughput, table structure accuracy (`TEDS`), downstream RAG accuracy, and TCO model comparing 4 document ingestion pipelines across a `10,000,000`-page enterprise PDF corpus.

## Run
```bash
python experiments/document_ai_pdf_ocr_parsing_mineru_docling_olmocr_vlm_layout_selection/run.py
python experiments/document_ai_pdf_ocr_parsing_mineru_docling_olmocr_vlm_layout_selection/run.py --check
```
