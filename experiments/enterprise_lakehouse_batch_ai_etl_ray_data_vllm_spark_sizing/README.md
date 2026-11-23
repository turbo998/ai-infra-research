# Enterprise Lakehouse Batch AI ETL (`50M Docs/Day`): `Spark HTTP UDF` vs. `Fabric AI Functions / Databricks ai_query` vs. `Ray Data + In-Process vLLM FP8` (`#532`)

- **Chinese FAQ**: [`../../docs/components/enterprise-lakehouse-batch-ai-etl-ray-data-vllm-spark-sizing-faq.md`](../../docs/components/enterprise-lakehouse-batch-ai-etl-ray-data-vllm-spark-sizing-faq.md)
- **English FAQ**: [`../../docs/components/enterprise-lakehouse-batch-ai-etl-ray-data-vllm-spark-sizing-faq.en.md`](../../docs/components/enterprise-lakehouse-batch-ai-etl-ray-data-vllm-spark-sizing-faq.en.md)
- **Interactive H5 Lab**: [`../../docs/h5/lakehouse-batch-ai-etl-ray-data-vllm-spark-lab.html`](../../docs/h5/lakehouse-batch-ai-etl-ray-data-vllm-spark-lab.html)
- **Interactive Sizer**: [`../../docs/reports/enterprise-lakehouse-batch-ai-etl-ray-data-vllm-spark-sizing-sizer.html`](../../docs/reports/enterprise-lakehouse-batch-ai-etl-ray-data-vllm-spark-sizing-sizer.html)
- **7-Card Report**: [`../../docs/reports/enterprise-lakehouse-batch-ai-etl-ray-data-vllm-spark-sizing-cards.html`](../../docs/reports/enterprise-lakehouse-batch-ai-etl-ray-data-vllm-spark-sizing-cards.html)

## Reproduce
```bash
python experiments/enterprise_lakehouse_batch_ai_etl_ray_data_vllm_spark_sizing/run.py
python -m pytest tests/test_enterprise_lakehouse_batch_ai_etl_ray_data_vllm_spark_sizing.py -q
```
