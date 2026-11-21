# How Many GPUs to Fit & Run an LLM? 3-Watermark Sizing (HBM Capacity × Bandwidth TPOT × Compute/Concurrency) Benchmark

Deterministic quantitative evaluation model for [`docs/components/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-faq.md`](../../docs/components/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-faq.md) and [`.en.md`](../../docs/components/llm-gpu-capacity-sizing-three-watermarks-memory-bandwidth-compute-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_gpu_capacity_sizing_three_watermarks_memory_bandwidth_compute_selection/run.py
python experiments/llm_gpu_capacity_sizing_three_watermarks_memory_bandwidth_compute_selection/run.py --check
python -m unittest tests/test_llm_gpu_capacity_sizing_three_watermarks_memory_bandwidth_compute_selection.py
```
