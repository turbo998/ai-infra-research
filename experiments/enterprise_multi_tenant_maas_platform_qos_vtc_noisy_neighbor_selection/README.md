# Enterprise Multi-Tenant MaaS Platform Architecture: 3-Tier QoS Isolation, Virtual Token Fair Queueing (VTC) & Noisy-Neighbor Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-selection-faq.md`](../../docs/components/enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-multi-tenant-maas-platform-qos-vtc-noisy-neighbor-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_multi_tenant_maas_platform_qos_vtc_noisy_neighbor_selection/run.py
python experiments/enterprise_multi_tenant_maas_platform_qos_vtc_noisy_neighbor_selection/run.py --check
python -m unittest tests/test_enterprise_multi_tenant_maas_platform_qos_vtc_noisy_neighbor_selection.py
```
