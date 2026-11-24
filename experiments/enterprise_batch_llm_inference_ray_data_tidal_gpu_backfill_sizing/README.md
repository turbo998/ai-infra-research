# Experiment #588: Enterprise Batch LLM Inference & Nightly Tidal GPU Backfill Sizer

Reproduces the `4.22x` per-GPU throughput boost (`4,850` vs. `1,150 tok/s/H200`) in prefix-sorted offline batch mode, `4.77-hour` completion time for `500M` tokens across `6` idle `H200` GPUs (`838M` max capacity per 8-hour night), and `97.5%` TCO reduction (`$285/mo` vs. `$11,250/mo` Cloud Batch API) (`docs/components/enterprise-batch-llm-inference-ray-data-tidal-gpu-backfill-sizing-faq.md`).
