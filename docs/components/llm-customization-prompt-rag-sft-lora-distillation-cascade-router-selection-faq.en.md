# LLM Customization Path & Distillation Selection Guide: `Prompt + Prefix Caching` vs. `Hybrid RAG` vs. `LoRA SFT` vs. `RAFT Teacher-Student Distillation & 82/18 Cascade Router`

> **Language**: [中文](./llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-faq.md) | **English (Current)**  
> **Audience**: Cloud & AI Infrastructure Solution Architects and Enterprise GenAI Leads  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)

![LLM Customization Prompt+Cache vs RAG vs LoRA SFT vs Distillation Cascade Summary](../../assets/svg/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary

> **Bottom Line**: **`RAG` and `Fine-Tuning (SFT / Distillation)` solve orthogonal problems: `Hybrid RAG` supplies dynamic, access-controlled external facts (`what to know`), while `SFT / Distillation` teaches domain schemas, DSL syntax, and reasoning patterns (`how to act`). Never use SFT alone to memorize volatile enterprise documents (`34.2%` hallucination rate and zero RBAC enforcement!). For static corpora `<=64K–128K` tokens, use `Long-Context Prompting + Prefix Caching` (`90%` cache hit). For high-volume production (`60M` queries/month), pair `Hybrid RAG` with a `RAFT`-distilled `8B FP8 Multi-LoRA` student model and an `82/18 Confidence Cascade Router` (`82%` handled by the `8B` student in `320ms`, `18%` escalated to the `671B` Teacher)—achieving `96.2%` task accuracy (`+0.8 pts` above pure Teacher), `510ms` P95 E2EL (`5.6x` faster), and `-79.4%` monthly TCO (`$248K -> $51K/mo`)!**

- **Interactive Customization ROI & Cascade Sizer**: [Open Interactive Sizer (`llm-customization-prompt-rag-sft-lora-distillation-cascade-router-sizer.html`)](../reports/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-sizer.html)
- **7-Card 4:5 Visual Report**: [Open 7-Card Report (`llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-cards.html`)](../reports/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-cards.html)

---

## 2. 📊 4-Tier Customization Comparison Matrix (`60M Queries / Month`)

| Metric | Tier 1: `SFT-Only` for Volatile Facts (No RAG) | Tier 2: 100% Frontier `671B Teacher + RAG` | Tier 3: Zero-Shot `8B Student + RAG` | Tier 4 (Recommended): `RAFT Distilled 8B Multi-LoRA + RAG + 82/18 Cascade` |
| :--- | :--- | :--- | :--- | :--- |
| **Domain Task Accuracy** | `65.8%` ✕ | `95.4%` ✓ | `79.2%` ✕ | **`96.2%` ✓ (`+0.8 pts`)** |
| **Factual Hallucination Rate** | `34.2%` ✕ | `2.4%` | `8.6%` | **`1.6%` ✓** |
| **P95 End-to-End Latency (`E2EL`)** | `620 ms` | `2,850 ms` ✕ | `380 ms` ✓ | **`510 ms` ✓ (`5.6x` faster)** |
| **Monthly `60M`-Query TCO** | `$195,000 / mo` | `$248,000 / mo` | `$42,000 / mo` | **`$51,000 / mo` ✓ (`-79.4%`)** |

---

## 3. 🛠️ Production Readiness Checklist

- [ ] **Never Use SFT to Memorize Dynamic Enterprise Facts**: Use `Hybrid RAG` for volatile facts and `RAFT / SFT` for domain reasoning and output structure.
- [ ] **Serve Multiple Domain Adapters on One Shared Backbone via `vLLM Multi-LoRA`** paired with an `82/18` logprob-confidence cascade router.
