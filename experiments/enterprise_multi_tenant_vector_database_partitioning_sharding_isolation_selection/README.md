# Enterprise Multi-Tenant Vector Database Isolation & Sharding (`5,000 Tenants / 100M Vectors`) Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-multi-tenant-vector-database-partitioning-sharding-isolation-selection-faq.md`](../../docs/components/enterprise-multi-tenant-vector-database-partitioning-sharding-isolation-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-multi-tenant-vector-database-partitioning-sharding-isolation-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_multi_tenant_vector_database_partitioning_sharding_isolation_selection/run.py
python experiments/enterprise_multi_tenant_vector_database_partitioning_sharding_isolation_selection/run.py --check
python -m unittest tests/test_enterprise_multi_tenant_vector_database_partitioning_sharding_isolation_selection.py
```
