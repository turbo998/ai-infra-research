# LLM Customization Selection: Prompt+Cache vs. Hybrid RAG vs. LoRA SFT vs. Teacher-Student Distillation & Cascade Router

Deterministic quantitative evaluation model for [`docs/components/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-faq.md`](../../docs/components/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-faq.md) and [`.en.md`](../../docs/components/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_customization_prompt_rag_sft_lora_distillation_cascade_router_selection/run.py
python experiments/llm_customization_prompt_rag_sft_lora_distillation_cascade_router_selection/run.py --check
python -m unittest tests/test_llm_customization_prompt_rag_sft_lora_distillation_cascade_router_selection.py
```
