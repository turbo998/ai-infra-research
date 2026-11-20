# Enterprise AI Model Storage, P2P Weight Distribution & 32-Node GPU Cold-Start Benchmark

Deterministic quantitative evaluation model for [`docs/components/enterprise-ai-model-storage-p2p-distribution-nvme-cache-cold-start-selection-faq.md`](../../docs/components/enterprise-ai-model-storage-p2p-distribution-nvme-cache-cold-start-selection-faq.md) and [`.en.md`](../../docs/components/enterprise-ai-model-storage-p2p-distribution-nvme-cache-cold-start-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/enterprise_ai_model_storage_p2p_distribution_nvme_cache_cold_start_selection/run.py
python experiments/enterprise_ai_model_storage_p2p_distribution_nvme_cache_cold_start_selection/run.py --check
python -m unittest tests/test_enterprise_ai_model_storage_p2p_distribution_nvme_cache_cold_start_selection.py
```
