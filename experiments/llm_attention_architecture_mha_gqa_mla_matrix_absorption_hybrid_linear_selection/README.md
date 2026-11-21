# LLM Attention Architecture Evolution: MHA vs. GQA-8 vs. MLA (Matrix Absorption & FlashMLA) vs. Hybrid Linear Attention

Deterministic quantitative evaluation model for [`docs/components/llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-selection-faq.md`](../../docs/components/llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-selection-faq.md) and [`.en.md`](../../docs/components/llm-attention-architecture-mha-gqa-mla-matrix-absorption-hybrid-linear-selection-faq.en.md).

## Provenance
- **Label**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

## Run & Verify
```bash
python experiments/llm_attention_architecture_mha_gqa_mla_matrix_absorption_hybrid_linear_selection/run.py
python experiments/llm_attention_architecture_mha_gqa_mla_matrix_absorption_hybrid_linear_selection/run.py --check
python -m unittest tests/test_llm_attention_architecture_mha_gqa_mla_matrix_absorption_hybrid_linear_selection.py
```
