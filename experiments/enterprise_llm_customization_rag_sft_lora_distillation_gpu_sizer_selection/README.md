# Enterprise LLM Customization Path (`Prompt+RAG` vs. `SFT/Multi-LoRA` vs. `Distillation`) & Cloud GPU SKU Sizing (`R473`)

Deterministic evaluation suite comparing four enterprise LLM customization and GPU hardware sizing architectures across `20` enterprise business units (`250 QPS` aggregate throughput, `80%` structured extraction/classification/SQL, `20%` complex multi-hop reasoning):

1. **Tier 1**: Blind Full-Weight Fine-Tuning (`70B SFT`) for Dynamic Knowledge across 20 Isolated Clusters (`62.4%` fresh-fact accuracy, `1,480ms` P99 latency, `64x H100`, `$245,000/mo` TCO)
2. **Tier 2**: Giant Few-Shot Prompt (`6,000 tokens`) + RAG on 70B Only (`89.5%` accuracy, `84.2%` schema adherence, `1,820ms` P99 latency, `32x H100`, `$142,000/mo` TCO)
3. **Tier 3**: RAG + Shared `70B Multi-LoRA` Pool (`1x 70B Base + 20 Hot-Swapped LoRA Adapters` on `8x H200`) (`95.8%` accuracy, `98.5%` schema adherence, `640ms` P99 latency, `$54,000/mo` TCO)
4. **Tier 4**: Hybrid RAG + Task-Distilled `8B` Student on `8x L40S` (80% Traffic) + Shared `70B Multi-LoRA` on `4x H200` (20% Traffic) (`96.6%` accuracy, `99.1%` schema adherence, `245ms` P99 latency, `$28,500/mo` TCO)

## Usage

```bash
python experiments/enterprise_llm_customization_rag_sft_lora_distillation_gpu_sizer_selection/run.py
python experiments/enterprise_llm_customization_rag_sft_lora_distillation_gpu_sizer_selection/run.py --check
python -m unittest tests/test_enterprise_llm_customization_rag_sft_lora_distillation_gpu_sizer_selection.py
```
