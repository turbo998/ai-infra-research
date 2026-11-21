# Prefill-Decode (PD) Disaggregation vs. Colocated Chunked Prefill: Boundary Thresholds, KV Transfer Math & Engine Benchmark

Deterministic quantitative evaluation model for [`docs/components/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-faq.md`](../../docs/components/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-faq.md) and [`.en.md`](../../docs/components/prefill-decode-disaggregation-vs-colocated-chunked-prefill-boundary-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection/run.py
python experiments/prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection/run.py --check
python -m unittest tests/test_prefill_decode_disaggregation_vs_colocated_chunked_prefill_boundary_selection.py
```
