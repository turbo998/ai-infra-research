# [导航索引] 大模型推理引擎选型与 Prefill-Decode (PD) 分离架构 FAQ（已拆分为两个独立单专题）

> 🌐 **Language / 语言**: **中文 (Chinese)** | **[English (`llm-inference-engine-pd-disaggregation-selection-faq.en.md`)](llm-inference-engine-pd-disaggregation-selection-faq.en.md)**
> 🧭 **单专题拆分说明**：本页仅作为旧复合链接的轻量导航索引，正文已拆分为以下 **2 篇独立单决策六段式 FAQ**：

1. **专题 1A：单机/同机混部 LLM 推理引擎选型（`vLLM` vs. `SGLang` vs. `TensorRT-LLM`）**
   - 中文：[`docs/components/llm-inference-engine-selection-faq.md`](llm-inference-engine-selection-faq.md) | English：[`docs/components/llm-inference-engine-selection-faq.en.md`](llm-inference-engine-selection-faq.en.md)
2. **专题 1B：跨池 Prefill-Decode (PD) 分离架构选型（同机混部 vs. `NVIDIA Dynamo + Mooncake`）**
   - 中文：[`docs/components/pd-disaggregation-selection-faq.md`](pd-disaggregation-selection-faq.md)（同 [`pd-disaggregation-kv-offload-selection-faq.md`](pd-disaggregation-kv-offload-selection-faq.md)） | English：[`docs/components/pd-disaggregation-selection-faq.en.md`](pd-disaggregation-selection-faq.en.md)
3. **关联算例与 7-Card 报告**：[`experiments/llm_inference_engine_pd_selection/results.json`](../../experiments/llm_inference_engine_pd_selection/results.json) · [`docs/reports/llm-inference-engine-pd-selection-cards.html`](../reports/llm-inference-engine-pd-selection-cards.html)
