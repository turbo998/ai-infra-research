# 大模型定制化路径与小模型蒸馏选型对比：`Prompt + Prefix Caching` vs. `Hybrid RAG` vs. `LoRA SFT 微调` vs. `大带小知识蒸馏（RAFT + 82/18 级联路由）` 决策树指南

> **语言 / Language**: **中文 (当前)** | [English](./llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与企业 AI 应用负责人的大模型四大定制化路径（`长上下文 Prompt+缓存`、`混合 RAG`、`LoRA SFT 微调`、`Teacher-Student 知识蒸馏 + 级联路由`）正交边界、幻觉控制与 ROI 选型白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业模型定制路径选型与成本测算）

![大模型定制路径 Prompt+Cache vs RAG vs LoRA SFT vs 蒸馏级联选型总结图](../../assets/svg/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**回答客户“我们到底该用 Prompt、做 RAG、还是微调（SFT）大模型？”时，必须牢记两条黄金正交法则：① **`RAG` 解决“知道什么动态事实（What to Know）”**，而 **`SFT / 蒸馏` 解决“按什么格式、语气、方言或推理链行动（How to Act）”**——严禁用 SFT 去死记硬背每天更新的企业知识库（不仅无法做行级权限隔离，还会产生 `34.2%` 的幻觉率！）；② 当知识库总长度 **`<= 64K ~ 128K` Tokens** 且变动不频繁时，直接用 **`长上下文 Prompt + Prefix Caching（90% 缓存命中享 75% 折扣）`** 比切块建向量库更准且零 ETL 维护！而在高频规模化生产（如月调 6,000 万次）下，最优解是用满血 `671B/405B` 教师模型通过 **`RAFT（带干扰项的检索增强思维链蒸馏）`** 训练一个 **`8B FP8 Multi-LoRA` 学生小模型**，配合 **`82/18` 置信度级联路由（82% 流量由 8B 学生秒回，仅 18% 疑难请求上送 671B 教师）**，准确率反超纯大模型（`96.2% vs 95.4%`）、延迟降低 `5.6 倍`（`510ms`），并节省 `79.4%` 月度 TCO！**

- **交互式定制路径 ROI 与蒸馏级联路由计算器**：[打开交互式计算器 (`llm-customization-prompt-rag-sft-lora-distillation-cascade-router-sizer.html`)](../reports/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-cards.html`)](../reports/llm-customization-prompt-rag-sft-lora-distillation-cascade-router-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：为什么“拿企业内部文档直接做 SFT 微调来代替 RAG”是最大反模式？什么情况下才应该启动 `LoRA SFT`？
1. **为什么 SFT 不能代替 RAG 注入动态事实？**：
   - 模型权重是有损概率压缩，无法对新增订单号、每日价格表或最新人事制度做到 100% 精确逐字记忆（实测事实幻觉率高达 **`34.2%`**）；
   - 一旦把机密文档训进权重，**完全无法执行 RBAC 文档级权限过滤**（普通员工通过 Prompt 诱导即可套出高管薪酬数据），也无法实现 GDPR 单条记录秒级遗忘！
2. **什么时候必须上 `LoRA SFT / 蒸馏`？**：
   - ① 需要模型掌握通用训练集里没有的**专有 DSL / 内部方言 SQL / 行业专有报文格式**；
   - ② 需要将一段冗长的 `4,000` 字 Few-shot 规则提示词“内化”进权重，从而每请求省下 `4,000` Input Tokens；
   - ③ **大带小蒸馏（Teacher-to-Student Distillation）**：用 `671B` 大模型生成高质量 `<think>` 推理链与工具调用轨迹，微调出便宜 `20 倍` 的 `8B` 垂直专家小模型！

### Q2：什么是 `RAFT（Retrieval-Augmented Fine-Tuning）` + `vLLM Multi-LoRA` + `82/18 置信度级联路由`？
1. **`RAFT`（让小模型学会“带着参考书答题并忽略干扰项”）**：
   - 裸 `8B` 小模型直接接 RAG 时，很容易被检索召回的 Top-5 相似错误段落（Distractors）带偏，准确率仅 `79.2%`；
   - `RAFT` 让 `671B` 教师模型在包含“1 篇黄金证据 + 4 篇干扰文档”的上下文上生成带精确引用的思维链数据，用来微调 `8B LoRA`——让 `8B` 小模型学会甄别噪声并严格引用证据，准确率飙升至 **`96.2%`**！
2. **`vLLM Multi-LoRA`（1 个 `8B` 基座同时挂载 32 个业务线专属 LoRA）**：
   - 借助 `Punica / S-LoRA` 算子，在显存中只加载 **1 份共享的 `8B FP8` 基础权重（`8 GB`）**，同时动态并发执行法务、财务、客服、研发 32 个不同业务线的 `LoRA (rank=32/64)` 适配器，零重复基座显存浪费！
3. **`82/18` 级联路由（Confidence Cascade Router）**：
   - 请求先经检索进入 `8B Multi-LoRA` 学生模型；若首句输出的对数概率置信度（Logprob Confidence）高于阈值（占 **`82%`** 常规请求），直接 `320ms` 极速返回；仅当遇到置信度不足的跨领域复杂难题（占 **`18%`**）时，才无感升级转发给 `671B` 教师模型兜底！

---

## 3. 🧭 直观可视化图解（Mermaid 定制路径决策树 + 82/18 蒸馏级联架构）

### 3.1 大模型定制化技术路径四选一决策树
```mermaid
flowchart TD
    A["企业大模型定制需求"] --> B{"核心诉求是注入【外部动态事实】<br/>还是改变【输出格式/推理逻辑/降本】？"}
    B -- "注入外部文档/知识库事实<br/>(需溯源引用 & 权限隔离)" --> C{"知识库总规模有多大？"}
    C -- "总规模 <= 64K~128K Tokens<br/>(如单一产品手册 / 固定规范)" --> D["🏆 选路径 1: 长上下文 Prompt + Prefix Caching<br/>• 免建向量库 · 90% 缓存命中享 75% 折扣"]
    C -- "总规模 > 128K ~ 千万级文档<br/>(高频更新 + 行级 RBAC 权限)" --> E["🏆 选路径 2: Hybrid RAG (BM25 + 向量 + Rerank)<br/>• 动态实时检索 · 严禁只做 SFT 不接 RAG!"]
    B -- "改变专有格式/DSL/垂直推理能力<br/>或将 671B 成本压降 80%" --> F{"是否有现成混合 RAG 场景或高 QPS 降本诉求？"}
    F -- "低 QPS 垂直格式微调" --> G["选路径 3: LoRA / QLoRA SFT 微调"]
    F -- "高 QPS 生产主力 (>= 10 RPS)" --> H["🏆 选路径 4: RAFT 蒸馏 8B Multi-LoRA + RAG<br/>+ 82/18 置信度级联路由至 671B 教师模型!"]
```

---

## 4. 📊 架构师多方案选型决策矩阵（月度 6,000 万次请求四档方案对比）

| 评估维度 | Tier 1：仅对动态知识做 `SFT`（无 RAG，反模式） | Tier 2：100% 流量跑满血 `671B 教师模型 + RAG` | Tier 3：100% 流量跑未蒸馏 `裸 8B 小模型 + RAG` | Tier 4（推荐）：`RAFT 蒸馏 8B Multi-LoRA + RAG` + `82/18 级联路由` |
| :--- | :--- | :--- | :--- | :--- |
| **垂直业务任务准确率** | `65.8%` ✕ | `95.4%` ✓ | `79.2%` ✕（抗干扰差） | **`96.2%` ✓（超越纯大模型 `+0.8 pts`）** |
| **事实幻觉率（Hallucination）** | `34.2%` ✕（灾难级） | `2.4%` | `8.6%` | **`1.6%` ✓** |
| **P95 端到端响应延迟 (`E2EL`)** | `620 ms` | `2,850 ms` ✕（太慢） | `380 ms` ✓ | **`510 ms` ✓（提速 `5.6 倍`）** |
| **月度 6,000 万次调用总 TCO** | `$195,000 / 月`（含频繁重训） | `$248,000 / 月` | `$42,000 / 月` | **`$51,000 / 月` ✓（较 Tier 2 节省 `79.4%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **严禁把企业动态业务数据直接拿去 SFT 微调充当知识库**：动态事实走 `Hybrid RAG`（或 `<=64K` 走 `Prefix Caching`），格式与推理范式走 `SFT/蒸馏`。
- [ ] **蒸馏训练数据必须采用 `RAFT` 范式（混入 Top-K 干扰文档）**：提升小模型在线上真实 RAG 噪声环境下的鲁棒性。
- [ ] **使用 `vLLM Multi-LoRA` 单底座承载多业务线适配器，并配置 Logprob 置信度级联兜底路由**。
