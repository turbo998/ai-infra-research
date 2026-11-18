# Enterprise Data & AI Lakehouse Platform Selection: Microsoft Fabric (Direct Lake), Databricks & Snowflake Evaluator

Deterministic evaluator comparing four enterprise Data & AI Lakehouse platform architectures across 1.5 PB of active storage, 4,500 daily Spark/ML jobs, and 12,000 business BI and AI Copilot users:
- **Tier 1**: Legacy Siloed Multi-Service Stack (`ADF + ADLS + Dedicated DW + Power BI Import` `3.2x` data copy, `180.0 min` freshness lag, `4.8 s` P95 latency, `$285,000/mo`)
- **Tier 2**: Monolithic Proprietary Cloud DW Used for All Unstructured ETL, ML & 12K-User BI (`2.1x` copy, `45.0 min` lag, `2.2 s` P95 latency, `$340,000/mo`)
- **Tier 3**: Pure Spark Lakehouse + Power BI `DirectQuery` over JDBC/Thrift (`1.4x` copy, `15.0 min` lag, `6.4 s` P95 latency, `$178,000/mo`)
- **Tier 4**: Open Zero-Copy Lakehouse (`OneLake Delta/Iceberg UniForm` + `Azure Databricks Photon` Heavy Spark/ML + `Microsoft Fabric Direct Lake` V-Order & `Fabric IQ` Semantic Layer) (`1.0x` single copy, `1.5 min` freshness lag, `0.45 s` P95 latency, `$92,000/mo`)

## Usage
```bash
python experiments/enterprise_data_platform_fabric_databricks_snowflake_lakehouse_selection/run.py
python experiments/enterprise_data_platform_fabric_databricks_snowflake_lakehouse_selection/run.py --check
```
