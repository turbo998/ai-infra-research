# High-Throughput Offline Batch Inference & Document Extraction: Prefix/Length Sorting, Max-Batch Roofline & Spot GPU Benchmark

Deterministic quantitative evaluation model for [`docs/components/offline-batch-inference-document-extraction-length-bucketing-spot-tco-selection-faq.md`](../../docs/components/offline-batch-inference-document-extraction-length-bucketing-spot-tco-selection-faq.md) and [`.en.md`](../../docs/components/offline-batch-inference-document-extraction-length-bucketing-spot-tco-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/offline_batch_inference_document_extraction_length_bucketing_spot_tco_selection/run.py
python experiments/offline_batch_inference_document_extraction_length_bucketing_spot_tco_selection/run.py --check
python -m unittest tests/test_offline_batch_inference_document_extraction_length_bucketing_spot_tco_selection.py
```
