# Enterprise GPU Sharing & Tidal Colocation Selection: Hardware MIG vs. CUDA MPS vs. Time-Slicing vs. HAMi/Kueue Tidal Benchmark

Deterministic quantitative evaluation model for [`docs/components/gpu-sharing-mig-mps-timeslicing-hami-tidal-colocation-selection-faq.md`](../../docs/components/gpu-sharing-mig-mps-timeslicing-hami-tidal-colocation-selection-faq.md) and [`.en.md`](../../docs/components/gpu-sharing-mig-mps-timeslicing-hami-tidal-colocation-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/gpu_sharing_mig_mps_timeslicing_hami_tidal_colocation_selection/run.py
python experiments/gpu_sharing_mig_mps_timeslicing_hami_tidal_colocation_selection/run.py --check
python -m unittest tests/test_gpu_sharing_mig_mps_timeslicing_hami_tidal_colocation_selection.py
```
