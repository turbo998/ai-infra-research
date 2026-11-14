# LLM Weight Serialization Format (`PyTorch Pickle` vs. `Safetensors` vs. `GGUF` vs. `Run:ai Streamer / Tensorizer GDS`) Selection (`llm_checkpoint_format_safetensors_gguf_tensorizer_selection`)

Deterministic Python 3.12 analytical evaluator for **Round 175**: comparing `Legacy PyTorch Pickle (.bin)` (Tier 1), `GGUF Single-File Edge/Hybrid Format` (Tier 2), `Standard HuggingFace Safetensors + OS mmap` (Tier 3), and `Pre-Sharded Safetensors / Tensorizer + Run:ai Streamer / GPUDirect Storage (GDS)` (Tier 4).

## Reproduction

```bash
python -m experiments.llm_checkpoint_format_safetensors_gguf_tensorizer_selection.run
python -m unittest tests/test_llm_checkpoint_format_safetensors_gguf_tensorizer_selection.py -q
```

## Provenance

- `type`: `synthetic_analytical_model_calibrated_to_published_storage_benchmarks`
- `independently_reproduced`: `false`
- `physical_gpu_cluster_measurement`: `not_measured`
