# LLM Inference Acceleration Panorama: TP/PP/DP/EP (DeepEP), Speculative Decoding, Prefix Caching & FP8 Benchmark

Deterministic quantitative evaluation model for [`docs/components/llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8-selection-faq.md`](../../docs/components/llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8-selection-faq.md) and [`.en.md`](../../docs/components/llm-inference-acceleration-tp-pp-dp-ep-speculative-prefix-fp8-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_inference_acceleration_tp_pp_dp_ep_speculative_prefix_fp8_selection/run.py
python experiments/llm_inference_acceleration_tp_pp_dp_ep_speculative_prefix_fp8_selection/run.py --check
python -m unittest tests/test_llm_inference_acceleration_tp_pp_dp_ep_speculative_prefix_fp8_selection.py
```
