# Experiment: Enterprise Multimodal Document AI & Complex PDF/Table Parsing Selection (`#411`)

Reproducible analytical model comparing `PyMuPDF/pdfplumber`, `Full-Page 72B VLM`, `Azure AI Document Intelligence`, and the `82/18 Hybrid Layout + Cropped Qwen2.5-VL-7B Pipeline (MinerU / IBM Docling)` across complex table RAG accuracy, GPU throughput (`pages/sec/GPU`), and cost per 1,000 pages.

- **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)
- **Run**: `python experiments/enterprise_document_ai_azure_doc_intelligence_vs_mineru_docling_vlm_tco_selection/run.py`
- **Verify**: `python experiments/enterprise_document_ai_azure_doc_intelligence_vs_mineru_docling_vlm_tco_selection/run.py --check`
