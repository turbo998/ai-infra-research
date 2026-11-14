# LLM Serving Engine CPU Scheduling Overhead, Python GIL & Zero-Overhead Async Multiprocess Architecture (`vLLM V1` / `SGLang`) Selection (`llm_serving_cpu_overhead_zeromq_async_scheduler_selection`)

Deterministic Python 3.12 analytical evaluator for **Round 178**: comparing `Monolithic Single-Process Loop (vLLM V0)` (Tier 1), `Multiprocess ZMQ IPC + Full Batch Metadata Rebuild` (Tier 2), `vLLM V1 Persistent Batch Delta State (CachedRequestData)` (Tier 3), and `Zero-Overhead Overlap Scheduler (SGLang Overlap / vLLM V1 Async Output + GPU Future Token Slot)` (Tier 4).

## Reproduction

```bash
python -m experiments.llm_serving_cpu_overhead_zeromq_async_scheduler_selection.run
python -m unittest tests/test_llm_serving_cpu_overhead_zeromq_async_scheduler_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_engine_benchmarks`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
