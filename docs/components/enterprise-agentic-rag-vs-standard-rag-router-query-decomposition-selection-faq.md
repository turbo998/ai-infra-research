# 企业级知识检索演进：标准单跳混合检索（`Single-Hop Hybrid RAG`）vs. 多步反思智能体检索（`Agentic RAG`）选型与延迟/成本拐点指南：什么时候单次 RAG 就够了，什么时候必须上 Agentic RAG，如何防止“查个分机号也跑 5 轮反思”导致 `9.4 秒` 卡顿与 `6 倍` 账单爆炸？

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-agentic-rag-vs-standard-rag-router-query-decomposition-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、企业知识管理平台负责人与 AI 应用架构师的“单跳混合 RAG vs. 多步 Agentic RAG 选型边界、`12ms` 查询复杂度路由器（`75% 快路径 / 25% 并行子查询拆解 DAG`）与 `CRAG` 相关性截断门禁（`max_steps=2`）”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection/results.json`](../../experiments/enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection/results.json)。

![企业级 Agentic RAG vs. 标准单跳混合 RAG 选型与路由架构总结图](../../assets/svg/enterprise-agentic-rag-vs-standard-rag-router-query-decomposition-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

随着 `Agentic RAG（智能体检索增强生成：自主规划 -> 改写查询 -> 多轮检索 -> 反思纠错 -> 跨源汇总）` 成为业界热词，很多企业研发团队一窝蜂地把原本好端端的知识库助手全量改造成了基于 `ReAct / LangGraph` 的多步反思循环。结果上线第二天就遭遇了两极分化的灾难：
1. **灾难一：全量死守“单跳朴素向量 RAG（Single-Hop Vector RAG）”（Tier 1，多跳对比/跨表问题准确率仅 `44.0%`）**：当高管问*“对比我们 Q3 欧洲区与 Q2 亚太区的云业务毛利率，并说明是哪几条新签大客户合同条款导致了差异？”*时，单次向量检索只能召回包含“Q3 欧洲”字眼的几个片段，漏掉亚太区数据和财务库指标，大模型当场胡编乱造！
2. **灾难二：把 100% 的用户提问全塞进无约束的 `ReAct Agentic RAG` 循环（Tier 2，综合 `P95` 响应延迟暴涨至 `9,450ms`，单次提问平均调用 `4.8 次` 大模型，月烧 `$318,000`）**：企业内部 **75% 的日常提问其实都是“VPN 报修电话是多少？”“差旅住宿标准多少钱？”这类单跳事实查询**！让它们也去跑“规划 -> 搜索 -> 反思 -> 改写 -> 再搜索”，不仅让员工干等近 10 秒，还白白烧掉 5 倍以上的 Token 预算！
3. **灾难三：当知识库里根本没有某篇新文件时，Agent 陷入“连续 6~8 轮改写关键词死循环”**：没有设置基于重排序得分（Reranker Score）的 **`CRAG（Corrective RAG）` 早停门禁**，导致查不到资料的请求反而消耗了全站最贵的算力！

在 **月均 3,000 万次企业知识查询（`30M Queries/月`，其中 `75%` 为单跳事实查询、`25%` 为跨文档对比/跨 SQL+文档多跳分析）** 的生产基准下，四档 RAG 架构的实测对决如下：

| 方案档位 | 核心检索与智能体编排架构 | 多跳/对比复杂问题准确率 | 全流量综合准确率 | 综合 `P95` 响应延迟 | 平均每问调用 LLM 次数 | 3,000 万次查询/月总成本 (`TCO`) | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：100% 单跳朴素向量 RAG (`Single-Hop Only`)** | 原话直接转单个向量查 Top-5，无拆解、无 SQL 联邦 | `44.0%` ✕ *(多实体对比全漏)* | `68.5%` ✕ | `680 ms` | `1.00 次` | `$48,000 / 月` | ⛔ 无法回答“A与B对比”及跨结构化/非结构化问题 |
| **🟨 Tier 2：100% 无约束 `ReAct Agentic RAG` (`3–6 轮串行`)** | 所有问题一律进自主反思循环 (`Plan->Search->Reflect`) | `91.5%` | `92.8%` | **`9,450 ms` ✕** *(慢如蜗牛)* | **`4.80 次` ✕** | **`$318,000 / 月` ✕** *(账单翻 6.6 倍)* | ⛔ 杀鸡用牛刀，75% 简单问题陪跑多轮串行循环 |
| **🟦 Tier 3：固定两跳改写 + 串行子查询 (`Serial Multi-Query`)** | 每问必调 LLM 改写 (+450ms)，再串行跑子查询 | `79.0%` | `85.4%` | `3,280 ms` ✕ | `2.40 次` | `$146,000 / 月` | ⚠️ 串行执行子查询太慢，且简单问题仍多花冤枉钱 |
| **🟩 Tier 4：`12ms` 复杂度路由器（`75% 单跳快路径` + `25% 并行子查询 DAG Agentic RAG`）+ `CRAG` 门禁** | **`12ms` 分类器分流：`75%` 走 `520ms` 单跳混合精排；`25%` 走一次拆解并行扇出 (`Vector+SQL+API`) + `max_steps=2` 早停** | **`94.6%` 🏆** *(并行全召回)* | **`95.8%` 🏆** | **`1,120 ms` 🏆** *(-88.1% vs Tier 2)* | **`1.35 次` 🏆** *(仅按需拆解)* | **`$64,000 / 月` 🏆** *(比全量 Agentic 省 79.9%)* | 🌟 **企业级生产知识底座“快慢双轨”唯一黄金标准** |

> 💡 **架构师一句话选型铁律（什么时候用 Standard RAG vs. Agentic RAG）**：
> - **永远不要做“二选一”，而是用 `12ms` 查询复杂度分类器做“快慢双轨分流（75 / 25 法则）”**：
>   1. **什么时候走标准单跳混合 RAG（占企业流量 `75%`）**：单一实体、单一制度、明确事实查找（如*“2026 年研发部报销额度是多少？”*）$\rightarrow$ 直接走 **`BM25 + 512d 向量 + L4 TEI Reranker` 单跳快路径（`520ms` 搞定，仅 `1.0 次` LLM 调用）**；
>   2. **什么时候才触发 Agentic RAG（占企业流量 `25%`）**：包含 **多实体/跨期对比（A vs B）**、**多跳因果推理** 或 **需同时拉取结构化数据库（Text-to-SQL / CRM API）与非结构化合同文档** 的复杂问题 $\rightarrow$ 且严禁用串行慢吞吞的 `ReAct` 一步步试探，必须用 **“一次性子问题拆解（Query Decomposition）+ `asyncio.gather` 多路并行检索扇出 + `CRAG` 重排得分门禁（`max_steps <= 2`）”**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击单向量召回盲区、9.4 秒串行假死与知识库缺失死循环 3 大痛点）

### Q1：为什么当用户问“对比 A 产品与 B 产品的故障率，并找出对应的供应商索赔条款”时，标准单跳 RAG 哪怕把 `Top-K` 调到 20 也会答错（准确率仅 `44.0%`）？
- **单一稠密向量的“语义重心偏移效应”与“跨模态数据源割裂”**：
  1. 当你把包含“A 产品 + B 产品 + 故障率统计表 + 供应商合同条款”的长句压缩成一个向量去检索时，向量余弦相似度会被字数最多的词主导（例如 Top-20 召回的全是 A 产品的说明书，B 产品和供应商合同在 Top-20 里占了 0 条）！
  2. 更关键的是，“故障率”存在数据仓库的 **SQL 表（Semantic Layer）** 里，而“索赔条款”存在法务部的 **PDF 向量库** 里。
  3. **Tier 4 的“一次性并行子查询拆解 DAG（Parallel Sub-Query Plan）”**：用轻量级 Planner 在 `180ms` 内将该复杂问题一次性拆解为 3 个正交子任务：`[SubQ1: SQL 查询 A/B 产品故障率, SubQ2: 向量检索 A 供应商索赔条款, SubQ3: 向量检索 B 供应商索赔条款]`，然后用 **`asyncio.gather` 三路同时并发执行**——耗时仅等于最慢单路（`~420ms`），且三个维度的证据 100% 齐备（多跳准确率直接从 `44.0%` 飙升至 **`94.6%`**）！

### Q2：很多团队用开源框架搭完 `Agentic RAG` 后，每问一个问题都要转圈 8~10 秒，怎么把 Agentic RAG 的延迟砍掉 80% 以上？
- **砍掉“串行 ReAct 思考圈”与“简单问题陪跑”的两把手术刀**：
  - **第一刀（入口 `12ms` 复杂度分流）**：在进入大模型前，用 `DeBERTa-v3-small / Qwen2.5-1.5B INT8` 分类器识别查询结构（单实体直查 vs. 多跳/比较/聚合）。把 **75% 的单跳问题直接旁路到 `520ms` 标准混合 RAG**，根本不进 Agent 状态机；
  - **第二刀（用 `Plan-and-Parallel-Execute DAG` 替代 `Serial ReAct`）**：传统 `ReAct` 是“想一步 -> 搜一次 A (1.5s) -> 看结果再想一步 -> 搜一次 B (1.5s) -> 再搜一次 C (1.5s)”的纯串行瀑布；改为 **有向无环图（DAG）并行扇出** 后，3~5 个子查询在同一毫秒并发打向向量库与 SQL 引擎，将复杂问题的端到端延迟从 `9,450ms` 压至 **`1,850ms`**（全流量加权 P95 仅 **`1,120ms`**）！

### Q3：如果员工问了一个知识库里根本不存在的新规定，怎么防止 Agentic RAG 在后台疯狂改写关键词搜索 6~8 轮，白白烧掉几万 Token？
- **`CRAG（Corrective RAG）` 重排置信度硬门禁 + `max_steps = 2` 铁律**：
  - 在第一轮并行检索返回 `Top-20` 切片后，直接读取 **`L4 HuggingFace TEI Cross-Encoder Reranker` 的最高相关性打分（`max_rerank_score ∈ [0, 1]`）**：
    - 若 **`max_rerank_score >= 0.65`**：证据充足，**0 次重试**，直接送入生成模型作答；
    - 若 **`0.35 <= max_rerank_score < 0.65`**：证据部分缺失，允许 **最多 `1 次` 扩写同义词补充检索（硬上限 `max_steps = 2`）**；
    - 若第 2 步后仍 **`< 0.35`**（或首轮得分 `< 0.15` 明显无关）：**立即熔断检索循环**，直接触发礼貌拒答（*“当前内部知识库尚未收录该政策”*）或路由至受控外部搜索工具——彻底消灭无意义的 6 轮死循环！

---

## ③ 🧭 直观可视化图解（快慢双轨选型决策树 + `75/25` 复杂度路由与并行 DAG 架构图 + 多跳并发检索时序图）

### 3.1 企业级标准单跳 RAG vs. Agentic RAG 选型决策树

```mermaid
flowchart TD
    A["企业级知识检索架构选型<br/>(月均 3,000 万次查询 | 75% 简单直查 + 25% 复杂多跳/跨源)"] --> B{"所有查询是否都走同一套检索流水线?"}
    B -- "100% 走单跳朴素向量 RAG (Tier 1)" --> C["⛔ 多跳对比/跨表问题准确率仅 44.0%！<br/>单向量无法同时召回多个对比实体与 SQL 指标"]
    B -- "100% 走无约束多轮串行 ReAct Agentic RAG (Tier 2)" --> D["⛔ P95 延迟高达 9,450ms，月烧 $31.8 万！<br/>75% 查电话号码的简单问题也被迫跑 4.8 次大模型"]
    B -- "入口部署 12ms 查询复杂度分类器 (Fast/Slow Dual-Track)" --> E{"分类器判定当前提问属于哪一类?"}
    E -- "75% 单实体/单制度事实直查 (Single-Hop)" --> F["🟦 快路径: 单跳混合检索 (BM25 + 512d 向量 + TEI 重排)<br/>(520ms 极速返回 | 仅 1.0 次 LLM 调用)"]
    E -- "25% 多实体对比 / 多跳因果 / 跨文档+SQL (Multi-Hop)" --> G["🟩 慢路径 (受控 Agentic): 一次性子问题拆解 + 并行 DAG 扇出 + CRAG 门禁 🏆<br/>(94.6% 多跳准确率 | max_steps=2 防死循环 | 综合省 79.9% TCO)"]
    style F fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `12ms` 快慢双轨路由 + 并行子查询拆解 DAG + `CRAG` 门禁总架构图

```mermaid
flowchart LR
    subgraph Ingress["1. 入口复杂度分流器 (12ms)"]
        Q["❓ 企业用户提问<br/>(3,000 万次/月)"] --> Cls["🧭 INT8 查询复杂度分类器<br/>识别: 单跳直查 vs 多跳对比/跨源"]
    end

    subgraph FastTrack["2. 快路径: 75% 单跳混合 RAG (520ms, 1.0x LLM)"]
        Cls -->|75% 简单直查| Fast["⚡ Hybrid BM25 + Vector + L4 TEI Rerank<br/>直接送入生成模型一次性作答"]
    end

    subgraph AgenticTrack["3. 慢路径: 25% 受控并行 DAG Agentic RAG (max_steps = 2)"]
        Cls -->|25% 多跳/对比/跨源| Decomp["🧩 一次性子查询拆解器 (Query Decomposer)<br/>生成并行有向无环图 (DAG)"]
        Decomp --> FanOut["🚀 asyncio.gather 多源并行扇出检索<br/>并发调用: 向量库(实体A) + 向量库(实体B) + 语义SQL/API"]
        FanOut --> CRAG["🛡️ CRAG 重排置信度门禁 (TEI Score >= 0.35)<br/>达标即合成答案；未达标最多改写 1 次否则诚实拒答"]
    end
```

### 3.3 跨文档 + 跨数据库多跳提问在 `1.4 秒` 内通过并行 DAG 完成检索与合成的时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 业务分析师
    participant Router as 12ms 复杂度路由器
    participant Planner as 子查询拆解器 (8B FP8)
    participant Fed as 并行联邦检索层 (Vector + Semantic SQL)
    participant CRAG as TEI Reranker & CRAG 门禁
    participant Synth as 综合生成模型 (70B / GPT-4o)

    User->>Router: 1. 提问：“对比 Q3 欧洲区与 Q2 亚太区云毛利率，并列出引发波动的新签合同免责条款”
    Router->>Planner: 2. 12ms 判定为【多跳跨源对比 (Multi-Hop)】，进入受控 Agentic DAG 通道
    Planner-->>Fed: 3. 180ms 一次性拆解出 3 个独立子查询：[SQL_Q3_EMEA, SQL_Q2_APAC, Vec_Contract_Clauses]
    par 三路子查询在 asyncio.gather 中同时并发执行（总耗时仅 380ms！）
        Fed->>Fed: 并发路1：向受管语义层查询 Q3 欧洲区毛利率 SQL
        Fed->>Fed: 并发路2：向受管语义层查询 Q2 亚太区毛利率 SQL
        Fed->>Fed: 并发路3：向合同向量库检索新签大客户免责条款 Top-20
    end
    Fed->>CRAG: 4. L4 TEI Reranker 在 34ms 内对合同切片精排，最高分 = 0.88 (>= 0.35 CRAG 门禁一次性通过！)
    CRAG->>Synth: 5. 将核验后的两组精确财务指标 + Top-5 合同条款送入生成模型
    Synth-->>User: 6. 端到端仅耗费 1.4 秒返回带精确数据表与合同原文引用的深度对比报告！
```

---

## ④ 📊 四档企业 RAG 与 Agentic RAG 架构全维度选型决策矩阵

| 评估维度 | 🟥 Tier 1：100% 单跳朴素 RAG | 🟨 Tier 2：100% 无约束 `ReAct Agentic` | 🟦 Tier 3：固定两跳串行改写 | 🟩 Tier 4：`75/25` 路由 + 并行 `DAG` + `CRAG` 门禁 |
|---|---|---|---|---|
| **多跳/跨源复杂问题准确率** | `44.0%` ✕ (单向量严重漏召回) | `91.5%` | `79.0%` | 🏆 **`94.6%`（正交子查询并行拆解 + SQL/向量联邦）** |
| **全流量综合回答准确率** | `68.5%` ✕ | `92.8%` | `85.4%` | 🏆 **`95.8%`（单跳快准 + 多跳深全 + CRAG 防幻觉）** |
| **全流量综合 `P95` 响应延迟** | `680 ms` | `9,450 ms` ✕ (串行跑 4.8 轮) | `3,280 ms` ✕ | 🏆 **`1,120 ms`（75% 走 `520ms` 快路径，25% 并行扇出）** |
| **平均每次提问调用 LLM 次数** | `1.00 次` | `4.80 次` ✕ (Token 消耗爆炸) | `2.40 次` | 🏆 **`1.35 次`（仅对 25% 复杂问题按需拆解，硬限 2 步）** |
| **3,000 万次查询/月总成本** | `$48,000 / 月` (但复杂问题不可用) | `$318,000 / 月` ✕ | `$146,000 / 月` | **`$64,000 / 月` 🏆（比全量 Agentic RAG 节省 79.9%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（月均 3,000 万次企业查询 / `75%` 单跳 + `25%` 多跳复杂分析）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级 Agentic RAG vs. 标准单跳混合 RAG 选型与成本拐点计算器](../reports/enterprise-agentic-rag-vs-standard-rag-router-query-decomposition-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-agentic-rag-vs-standard-rag-router-query-decomposition-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection/run.py --check
  python -m unittest tests/test_enterprise_agentic_rag_vs_standard_rag_router_query_decomposition_selection.py
  ```

### 5.2 核心量化对比图（多跳复杂问题准确率 % vs. 月度平台总成本千美元）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Single-Hop Vector RAG Only", "metric": "Multi-Hop Accuracy (%)", "val": 44.0 },
      { "tier": "Tier 1: Single-Hop Vector RAG Only", "metric": "Monthly TCO ($K)", "val": 48.0 },
      { "tier": "Tier 2: Unbounded ReAct Agentic RAG", "metric": "Multi-Hop Accuracy (%)", "val": 91.5 },
      { "tier": "Tier 2: Unbounded ReAct Agentic RAG", "metric": "Monthly TCO ($K)", "val": 318.0 },
      { "tier": "Tier 3: Static 2-Hop Serial Rewrite", "metric": "Multi-Hop Accuracy (%)", "val": 79.0 },
      { "tier": "Tier 3: Static 2-Hop Serial Rewrite", "metric": "Monthly TCO ($K)", "val": 146.0 },
      { "tier": "Tier 4: 75/25 Router + Parallel DAG", "metric": "Multi-Hop Accuracy (%)", "val": 94.6 },
      { "tier": "Tier 4: 75/25 Router + Parallel DAG", "metric": "Monthly TCO ($K)", "val": 64.0 }
    ]
  },
  "semantic_types": { "tier": "Category", "metric": "Name", "val": "Quantity" },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 achieves 94.6% multi-hop accuracy at $64K/mo (vs. $318K/mo for unbounded Agentic RAG)",
    "subtitle": "Multi-Hop Query Accuracy (%) vs. Monthly 30M-Query Platform TCO ($ Thousands)",
    "encodings": { "x": "tier", "y": "val", "group": "metric", "color": "metric" }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（上线前必查 5 项）

- [ ] **严禁将 100% 流量无脑送入多步 `ReAct Agentic RAG` 循环**：必须在入口处部署 `< 15ms` 的查询复杂度分类器，将占总量 70%~80% 的单实体直查请求保留在 `520ms` 单跳混合检索快路径上。
- [ ] **对 25% 复杂多跳问题采用“一次性子问题拆解（Query Decomposition）+ `asyncio.gather` 并行检索扇出”**：严禁让 Agent 以串行 `for` 循环方式一个接一个查子问题。
- [ ] **引入基于 Cross-Encoder Reranker 分数的 `CRAG（Corrective RAG）` 早停门禁（阈值 `0.35`）**：当精排最高分已达标时直接合成答案；当得分过低时最多允许 **1 次** 改写（`max_steps = 2`），仍低于阈值立即诚实拒答。
- [ ] **打通“非结构化文档向量库 + 结构化受管语义层（Text-to-SQL / REST API）”联邦子查询**：严禁把实时变化的 ERP/CRM 指标硬切成静态文本块塞进向量库。
- [ ] **对拆解出的多个子查询结果执行统一去重（Chunk ID Deduplication）与来源锚点标号**：确保最终汇总模型能够精确引用每一项对比数据背后的原始文档或 SQL 视图。
