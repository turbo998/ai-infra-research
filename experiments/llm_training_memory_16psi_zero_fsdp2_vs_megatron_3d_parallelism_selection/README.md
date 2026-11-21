# LLM Training Memory Anatomy (16P State + Activations): ZeRO-1/2/3 (FSDP2) vs. Megatron 3D Parallelism Benchmark

Deterministic quantitative evaluation model for [`docs/components/llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-selection-faq.md`](../../docs/components/llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-selection-faq.md) and [`.en.md`](../../docs/components/llm-training-memory-16psi-zero-fsdp2-vs-megatron-3d-parallelism-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_training_memory_16psi_zero_fsdp2_vs_megatron_3d_parallelism_selection/run.py
python experiments/llm_training_memory_16psi_zero_fsdp2_vs_megatron_3d_parallelism_selection/run.py --check
python -m unittest tests/test_llm_training_memory_16psi_zero_fsdp2_vs_megatron_3d_parallelism_selection.py
```
