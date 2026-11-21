# LLM Quantization Deep Dive: BF16 vs. Per-Tensor FP8 vs. Fine-Grained Block FP8 (1x128/128x128) vs. W4A16 Marlin vs. NVFP4

Deterministic quantitative evaluation model for [`docs/components/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-faq.md`](../../docs/components/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-faq.md) and [`.en.md`](../../docs/components/llm-quantization-fp8-blockwise-nvfp4-awq-marlin-accuracy-throughput-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_quantization_fp8_blockwise_nvfp4_awq_marlin_accuracy_throughput_selection/run.py
python experiments/llm_quantization_fp8_blockwise_nvfp4_awq_marlin_accuracy_throughput_selection/run.py --check
python -m unittest tests/test_llm_quantization_fp8_blockwise_nvfp4_awq_marlin_accuracy_throughput_selection.py
```
