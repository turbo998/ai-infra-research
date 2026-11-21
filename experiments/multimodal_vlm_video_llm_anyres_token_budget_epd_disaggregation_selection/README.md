# Multimodal VLM & Video-LLM Architecture: AnyRes Visual Token Budgeting & E-P-D (Encoder-Prefill-Decode) Disaggregation Benchmark

Deterministic quantitative evaluation model for [`docs/components/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-faq.md`](../../docs/components/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-faq.md) and [`.en.md`](../../docs/components/multimodal-vlm-video-llm-anyres-token-budget-epd-disaggregation-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/multimodal_vlm_video_llm_anyres_token_budget_epd_disaggregation_selection/run.py
python experiments/multimodal_vlm_video_llm_anyres_token_budget_epd_disaggregation_selection/run.py --check
python -m unittest tests/test_multimodal_vlm_video_llm_anyres_token_budget_epd_disaggregation_selection.py
```
