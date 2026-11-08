# [导航索引] 超大模型推理量化精度与 MoE 分布式并行选型 FAQ（已按“一篇一决策”拆分为两个独立单专题）

> 🌐 **Language / 语言**: **中文 (Chinese)** | **[English (`moe-quantization-parallelism-selection-faq.en.md`)](moe-quantization-parallelism-selection-faq.en.md)**
> 🧭 **单专题拆分说明**：为严格遵循本仓库标杆文档 [`ib-vs-ethernet-roce-selection-faq.md`](ib-vs-ethernet-roce-selection-faq.md) 的“一篇一独立决策”门禁，原复合页面已拆分为以下 **2 篇独立的中英双语六段式 FAQ 指南**（本页仅保留旧链接兼容与锚点迁移导航，不重复承载复合正文，亦不计为额外研究轮次）：

---

## 🔗 请前往拆分后的两篇独立单决策 FAQ 专题

### 1. 专题 2A：大模型推理量化精度选型（`BF16` vs. `W4A16 (AWQ/Marlin)` vs. `细粒度 FP8` vs. `硬件原生 FP4 (NVFP4 / MXFP4)`）
- 📄 **中文完整六段式指南**：[`docs/components/llm-quantization-selection-faq.md`](llm-quantization-selection-faq.md)
- 📄 **English Full 6-Part Guide**：[`docs/components/llm-quantization-selection-faq.en.md`](llm-quantization-selection-faq.en.md)
- 🎯 **核心解决决策**：何时用权重量化 `W4A16`（低并发 `45,100 tok/s`，`26.4 ms` P99 TPOT，`98.4%` 精度保持，`$0.71/1M`）、何时用 Hopper/MI300X/MI325X 细粒度块缩放 `FP8`（`92,200 tok/s`，`15.8 ms` P99 TPOT，`99.6%` 精度保持，`$0.35/1M`）、何时用 Blackwell 原生 `NVFP4` 或 MI355X 原生 `MXFP4`（`151,200 tok/s`，`11.2 ms` P99 TPOT，`98.9%` 精度保持，`$0.21/1M`），以及各 GPU 代际的硬件指令支持边界。

### 2. 专题 2B：超大 MoE 模型跨机并行策略选型（跨机纯 `TP=16` vs. `Attention DP + MoE EP + DeepEP + EPLB`）
- 📄 **中文完整六段式指南**：[`docs/components/moe-parallelism-selection-faq.md`](moe-parallelism-selection-faq.md)
- 📄 **English Full 6-Part Guide**：[`docs/components/moe-parallelism-selection-faq.en.md`](moe-parallelism-selection-faq.en.md)
- 🎯 **核心解决决策**：为什么跨节点纯 `TP=16` 会遭遇 AllReduce 延迟墙（`34,800 tok/s`，`34.8 ms` P99 TPOT，`$0.92/1M`），何时切换为 `Attention DP=8 + MoE EP=8 + DeepEP`（提升 `2.65x` 至 `92,200 tok/s`），以及为什么必须启用 `EPLB` 动态专家负载均衡（消除 Zipf `s=1.15` 热门专家偏斜，使吞吐从无 EPLB 的 `60,220 tok/s` 提升 `+53.1%` 至 `92,200 tok/s`，P99 TPOT 从 `25.1 ms` 降至 `15.8 ms` `-37.1%`）。

---

## 📊 关联封存算例与可视化资产
- 🧪 **封存确定性算例 (`synthetic`, `independently_reproduced=false`)**：[`experiments/moe_quantization_parallelism_selection/run.py`](../../experiments/moe_quantization_parallelism_selection/run.py) · [`experiments/moe_quantization_parallelism_selection/results.json`](../../experiments/moe_quantization_parallelism_selection/results.json)
- 🖼️ **一图流总结 SVG**：[`assets/svg/moe-quantization-parallelism-selection-faq-summary.svg`](../../assets/svg/moe-quantization-parallelism-selection-faq-summary.svg)
- 🃏 **7-Card 4:5 视觉评测报告**：[`docs/reports/moe-quantization-parallelism-selection-cards.html`](../reports/moe-quantization-parallelism-selection-cards.html)
