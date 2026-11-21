# AI Hypercomputer Control Plane (GKE Kueue TAS + JobSet + DWS vs. Slurm) & 3-Tier Storage Benchmark

Deterministic quantitative evaluation model for [`docs/components/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-faq.md`](../../docs/components/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-faq.md) and [`.en.md`](../../docs/components/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection/run.py
python experiments/ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection/run.py --check
python -m unittest tests/test_ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection.py
```
