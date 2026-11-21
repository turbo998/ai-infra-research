# AI Cluster Silent Data Corruption (SDC), Xid/ECC Fault Triage & Sub-60s Warm-Spare Resilient Training Benchmark

Deterministic quantitative evaluation model for [`docs/components/ai-cluster-silent-data-corruption-sdc-xid-ecc-resilient-training-selection-faq.md`](../../docs/components/ai-cluster-silent-data-corruption-sdc-xid-ecc-resilient-training-selection-faq.md) and [`.en.md`](../../docs/components/ai-cluster-silent-data-corruption-sdc-xid-ecc-resilient-training-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/ai_cluster_silent_data_corruption_sdc_xid_ecc_resilient_training_selection/run.py
python experiments/ai_cluster_silent_data_corruption_sdc_xid_ecc_resilient_training_selection/run.py --check
python -m unittest tests/test_ai_cluster_silent_data_corruption_sdc_xid_ecc_resilient_training_selection.py
```
