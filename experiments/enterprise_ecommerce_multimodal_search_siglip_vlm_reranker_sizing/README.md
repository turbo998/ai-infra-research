# Enterprise E-Commerce Visual Search & Conversational Shopping Guide (`100M SKUs · 5,000 Peak QPS`): `Direct 72B VLM` vs. `OCR + BM25` vs. `SigLIP-2 Dual-Tower ANN + Dynamic-Tile 7B VLM Reranker + Gated 70B Advisor` (`#536`)

- **Chinese FAQ**: [`../../docs/components/enterprise-ecommerce-multimodal-search-siglip-vlm-reranker-sizing-faq.md`](../../docs/components/enterprise-ecommerce-multimodal-search-siglip-vlm-reranker-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-ecommerce-multimodal-search-siglip-vlm-reranker-sizing-faq.en.md`](../../docs/components/enterprise-ecommerce-multimodal-search-siglip-vlm-reranker-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/ecommerce-multimodal-search-siglip-vlm-reranker-lab.html`](../../docs/h5/ecommerce-multimodal-search-siglip-vlm-reranker-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-ecommerce-multimodal-search-siglip-vlm-reranker-sizing-sizer.html`](../../docs/reports/enterprise-ecommerce-multimodal-search-siglip-vlm-reranker-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-ecommerce-multimodal-search-siglip-vlm-reranker-sizing-cards.html`](../../docs/reports/enterprise-ecommerce-multimodal-search-siglip-vlm-reranker-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_ecommerce_multimodal_search_siglip_vlm_reranker_sizing/run.py
python -m pytest tests/test_enterprise_ecommerce_multimodal_search_siglip_vlm_reranker_sizing.py -q
```
