# Enterprise E-Commerce Product Catalog Enrichment, Hierarchical Taxonomy & Multimodal Dedup Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-ecommerce-product-catalog-enrichment-taxonomy-vlm-dedup-selection-faq.md`](../../docs/components/enterprise-ecommerce-product-catalog-enrichment-taxonomy-vlm-dedup-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-ecommerce-product-catalog-enrichment-taxonomy-vlm-dedup-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_ecommerce_product_catalog_enrichment_taxonomy_vlm_dedup_selection/run.py
python experiments/enterprise_ecommerce_product_catalog_enrichment_taxonomy_vlm_dedup_selection/run.py --check
python -m unittest tests/test_enterprise_ecommerce_product_catalog_enrichment_taxonomy_vlm_dedup_selection.py
```
