# Speculative Decoding Deep Dive: Standalone Draft vs. Medusa vs. EAGLE-3 vs. Native MTP & Batch-Size Crossover Benchmark

Deterministic quantitative evaluation model for [`docs/components/speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-selection-faq.md`](../../docs/components/speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-selection-faq.md) and [`.en.md`](../../docs/components/speculative-decoding-draft-medusa-eagle3-mtp-acceptance-batch-boundary-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/speculative_decoding_draft_medusa_eagle3_mtp_acceptance_batch_boundary_selection/run.py
python experiments/speculative_decoding_draft_medusa_eagle3_mtp_acceptance_batch_boundary_selection/run.py --check
python -m unittest tests/test_speculative_decoding_draft_medusa_eagle3_mtp_acceptance_batch_boundary_selection.py
```
