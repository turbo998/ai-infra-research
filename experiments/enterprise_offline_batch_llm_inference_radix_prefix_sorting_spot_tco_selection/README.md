# Enterprise High-Throughput Offline Batch LLM Inference (`Lexicographical Radix Prefix Sorting`, `Length Bucketing` & `Spot/Tidal GPU` Sizer) (`#474`)

Reproducible analytical model comparing Online Pool Injection, 50%-Off Cloud Batch API, Unsorted High-Batch GPU Pools, and `Ray Data` + Lexicographical Radix Prefix Sorting + Output-Length Bucketing on Spot/Tidal GPUs (`$0.028 / 1M tokens`, `9,840 tok/s/GPU`).

- **Provenance**: `synthetic + anonymized_reported (independently_reproduced=false)`
- **Run**: `python experiments/enterprise_offline_batch_llm_inference_radix_prefix_sorting_spot_tco_selection/run.py --check`
