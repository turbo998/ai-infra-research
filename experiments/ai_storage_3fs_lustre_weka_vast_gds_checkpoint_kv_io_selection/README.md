# AI Cluster High-Performance Storage Selection: DeepSeek 3FS vs. Lustre/GPFS vs. WEKA/VAST vs. Object FUSE + GDS Benchmark

Deterministic quantitative evaluation model for [`docs/components/ai-storage-3fs-lustre-weka-vast-gds-checkpoint-kv-io-selection-faq.md`](../../docs/components/ai-storage-3fs-lustre-weka-vast-gds-checkpoint-kv-io-selection-faq.md) and [`.en.md`](../../docs/components/ai-storage-3fs-lustre-weka-vast-gds-checkpoint-kv-io-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/ai_storage_3fs_lustre_weka_vast_gds_checkpoint_kv_io_selection/run.py
python experiments/ai_storage_3fs_lustre_weka_vast_gds_checkpoint_kv_io_selection/run.py --check
python -m unittest tests/test_ai_storage_3fs_lustre_weka_vast_gds_checkpoint_kv_io_selection.py
```
