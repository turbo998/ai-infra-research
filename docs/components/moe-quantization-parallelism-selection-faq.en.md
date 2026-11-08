# [Navigation Portal] Large Model Quantization Precision & MoE Multi-Node Parallelism Selection FAQs (Split into Two Single-Decision Guides)

> 🌐 **Language / 语言**: **[中文 (`moe-quantization-parallelism-selection-faq.md`)](moe-quantization-parallelism-selection-faq.md)** | **English**
> 🧭 **Single-Decision Split Notice**: In accordance with our single-decision-per-FAQ standard modeled on [`ib-vs-ethernet-roce-selection-faq.en.md`](ib-vs-ethernet-roce-selection-faq.en.md), this legacy composite entry has been split into **two standalone bilingual 6-part FAQ guides**. This page serves solely as a lightweight backward-compatible navigation portal and does not duplicate the composite prose or count as an extra research round.

---

## 🔗 Standalone Single-Decision FAQ Guides

### 1. Topic 2A: LLM Inference Quantization Precision Selection (`BF16` vs. `W4A16 (AWQ/Marlin)` vs. `Fine-Grained FP8` vs. `Hardware-Native FP4 (NVFP4 / MXFP4)`)
- 📄 **English Full 6-Part Guide**: [`docs/components/llm-quantization-selection-faq.en.md`](llm-quantization-selection-faq.en.md)
- 📄 **中文完整六段式指南**: [`docs/components/llm-quantization-selection-faq.md`](llm-quantization-selection-faq.md)
- 🎯 **Decision Scope**: When to choose weight-only `W4A16` (`45,100 tok/s`, `26.4 ms` P99 TPOT, `98.4%` retention, `$0.71/1M`), fine-grained block-scaled `FP8` on Hopper H100/H200 & AMD MI300X/MI325X (`92,200 tok/s`, `15.8 ms` P99 TPOT, `99.6%` retention, `$0.35/1M`), or hardware-native `NVFP4` (Blackwell B200/GB200) / `MXFP4` (AMD MI355X CDNA4) (`151,200 tok/s`, `11.2 ms` P99 TPOT, `98.9%` retention, `$0.21/1M`).

### 2. Topic 2B: Large MoE Multi-Node Parallelism Selection (Cross-Node Pure `TP=16` vs. `Attention DP + MoE EP + DeepEP + EPLB`)
- 📄 **English Full 6-Part Guide**: [`docs/components/moe-parallelism-selection-faq.en.md`](moe-parallelism-selection-faq.en.md)
- 📄 **中文完整六段式指南**: [`docs/components/moe-parallelism-selection-faq.md`](moe-parallelism-selection-faq.md)
- 🎯 **Decision Scope**: Why cross-node pure `TP=16` hits an AllReduce latency wall (`34,800 tok/s`, `34.8 ms` P99 TPOT, `$0.92/1M`), when to switch to `Attention DP=8 + MoE EP=8 + DeepEP` (`2.65x` speedup to `92,200 tok/s`), and why `EPLB` dynamic hot-expert replication is mandatory under Zipf `s=1.15` routing skew (`60,220 -> 92,200 tok/s` `+53.1%` goodput, `25.1 -> 15.8 ms` `-37.1%` P99 TPOT).

---

## 📊 Sealed Experiment & Visual Artifacts
- 🧪 **Deterministic Evaluator (`synthetic`, `independently_reproduced=false`)**: [`experiments/moe_quantization_parallelism_selection/run.py`](../../experiments/moe_quantization_parallelism_selection/run.py) · [`experiments/moe_quantization_parallelism_selection/results.json`](../../experiments/moe_quantization_parallelism_selection/results.json)
- 🖼️ **Visual Summary SVG**: [`assets/svg/moe-quantization-parallelism-selection-faq-summary.svg`](../../assets/svg/moe-quantization-parallelism-selection-faq-summary.svg)
- 🃏 **7-Card 4:5 Benchmark Deck**: [`docs/reports/moe-quantization-parallelism-selection-cards.html`](../reports/moe-quantization-parallelism-selection-cards.html)
