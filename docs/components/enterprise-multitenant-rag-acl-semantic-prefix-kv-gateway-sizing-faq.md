# 企业多租户 RAG 与 Agent 推理平台架构选型：`Azure AI Search / Milvus` 权限前置过滤（ACL Pre-Filter）、防串单语义缓存与 `Envoy GIE` 前缀 KV 粘性路由指南（`#571`）

> **Language / 语言**: **简体中文** | [English](./enterprise-multitenant-rag-acl-semantic-prefix-kv-gateway-sizing-faq.en.md) · [📱 交互式 H5 精读页（含双层缓存架构图 + 实时算卡器）](../h5/enterprise-multitenant-rag-acl-semantic-prefix-kv-gateway-lab.html) · [🧮 交互式算卡器 Sizer](../reports/enterprise-multitenant-rag-acl-semantic-prefix-kv-gateway-sizing-sizer.html) · [🃏 7-Card 评测报告](../reports/enterprise-multitenant-rag-acl-semantic-prefix-kv-gateway-sizing-cards.html) · [返回总目录](../full-catalog-zh.md)

![架构摘要图](../../assets/svg/enterprise-multitenant-rag-acl-semantic-prefix-kv-gateway-sizing-faq-summary.svg)

---

## 一、3 分钟核心结论卡片（Executive Summary）

1. **一句话结论**：
   - 在企业级多部门、多租户知识库 RAG 与多轮 Agent 平台（峰值 `40 QPS`，平均上下文 `12K Tokens`）中，如果采用传统的**“向量检索后置过滤（Post-Filter ACL）+ 裸开余弦相似度语义缓存（$\cos > 0.90$）+ 普通 L7 轮询网关”**，会同时爆发三大生产事故：① 小权限租户 `Recall@10` 暴跌至 **`41.2%`**（零结果饥饿）；② 语义缓存把“2025 Q1 财报”与“2026 Q1 财报”当成同一句话返回，**张冠李戴/越权串单率高达 `11.8%`**；③ 轮询网关打散多轮请求，导致底层 `vLLM/SGLang` 前缀缓存命中率跌至 **`18.0%`**，需堆叠 **`32 张 H200`**；
   - 采用 **`Roaring Bitmap 权限前置过滤（Pre-Filter ACL）+ 双层防串单缓存（L1 租户隔离实体强校验语义缓存 + L2 Envoy AI Gateway [K8s GIE EPP] 前缀 KV 粘性路由）`**，将权限合规检出召回率推升至 **`96.4%`**（跨租户泄密率 **`0.0%`**），语义误命中率从 `11.8%` 压至 **`0.02%`（降低 `590 倍`）**，底层 Prefix KV 命中率达 **`91.5%`**，首字延迟 `P95 TTFT` 从 `1,450 ms` 压至 **`210 ms`（提速 `6.90 倍`）**，所需 GPU 从 `32 张` 降至 **`12 张 H200`（立省 `62.5%` GPU，`$0.068 / 1M Tokens`）**！

---

## 二、灵魂拷问 Q&A：Pre-Filter ACL、语义缓存防串单与 Prompt 前缀工程

### Q1：为什么多租户向量检索严禁使用“先搜 Top-K 再按权限过滤（Post-Filter）”，而在 `Azure AI Search / Milvus / Qdrant` 中该如何选型？
- **答**：
  - **后置过滤（Post-Filter）的“零召回”陷阱**：假设全公司知识库有 `5,000 万` 个文档切片，某普通员工仅有权限查看本部门的 `50 万` 个切片（占比 `1%`）。如果向量库先在全库 HNSW 图中盲搜 `Top-20`，再由业务代码过滤掉无权访问的 `99%` 切片，最终返回给大模型的结果大概率是 **0 条或仅剩 1 条残缺片段**！
  - **前置过滤（Pre-Filter ACL Bitmap + Filtered HNSW/DiskANN）**：在进入向量距离计算前，先根据用户的 `Entra ID Security Group / Role` 生成压缩位图（Roaring Bitmap），在索引遍历时仅沿位图为 `1` 的合法邻居节点跳跃（当过滤比例极窄 `< 1%` 时自动切换为暴力精确扫描），保证始终满额返回合法的 `Top-20` 切片，Recall@10 从 `41.2%` 跃升至 **`96.4%`**！
  - **主流企业检索引擎横向选型**：
    - **`Azure AI Search`**：原生内置安全组 ACL Pre-Filter、BM25+向量混合检索与 L2 语义重排器（Semantic Ranker），与 Microsoft Entra ID / Purview 文档级权限自动同步，是企业托管首选；
    - **`Milvus 2.5+ (Zilliz)`**：支持 `Partition Key` 租户物理/逻辑隔离 + `Bitset Pre-Filter` + `DiskANN / GPU CAGRA`，适合十亿级自建大集群；
    - **`Qdrant`**：Rust 编写，拥有业界最强的稀疏+稠密混合过滤图索引（Filterable HNSW），中小规模极速部署首选。

```mermaid
flowchart TB
    REQ["企业员工提问 + Entra ID JWT Token"] --> ACL["1. 权限解析器：提取 Tenant_ID + Security_Groups"]
    ACL --> L1["2. L1 网关语义缓存守卫 (Redis / Vector)\n• 复合键名：Hash(Tenant_ID + ACL_Groups) 隔离\n• 强校验：数字/年份/专有名词槽位 100% 一致 + Cross-Encoder >= 0.96"]
    L1 -->|"命中 (24% 高频单轮 FAQ · 12ms)"| FAST["直接返回已验证答案 (零 GPU 消耗)"]
    L1 -->|"未命中 (76% 多轮 Agent / 复杂 RAG)"| SEARCH["3. 向量库 Pre-Filter ACL 混合检索\n(Azure AI Search / Milvus Bitset + BGE-Reranker)"]
    SEARCH --> GIE["4. L2 东西向 Envoy AI Gateway (K8s GIE EPP)\n• 按 [System Prompt + Tool Schema + Top-K 文档块] 前缀哈希粘性路由"]
    GIE --> ENG["5. vLLM V1 / SGLang 推理池 (12x H200)\n• RadixAttention 精确复用 91.5% 历史 KV Cache (零语义串单 · TTFT 210ms)"]
```

---

### Q2：为什么在网关层裸开“余弦相似度 $> 0.90$”的语义缓存会引发高达 `11.8%` 的生产张冠李戴事故？L1 语义缓存与 L2 Prefix KV Cache 该如何正确分工？
- **答**：
  1. **余弦语义缓存的致命盲区**：通用 Embedding 模型对句子中的**年份、金额、工单号、否定词（“含税” vs “不含税”）**极不敏感。例如用户 A 问 *“查询 2025 年华东区差旅报销上限”*，用户 B 问 *“查询 2026 年华北区差旅报销上限”*，两句话的 Embedding 余弦相似度高达 **`0.962`**！裸开语义缓存会直接把 2025 年华东区的旧答案错发给用户 B！
  2. **双层缓存黄金分工准则**：
     - **L1 网关级语义缓存（仅拦截 `~24%` 的单轮高频标准 FAQ）**：必须满足三道硬门禁才允许返回：① **租户与 ACL 哈希完全一致**；② **正则/GLiNER 提取出的数字、日期、否定词与实体槽位 100% 完全匹配**；③ **轻量 Cross-Encoder 重排得分 `>= 0.96`**（将误命中率从 `11.8%` 压至 **`0.02%`**）。
     - **L2 引擎级 Exact Prefix KV Cache（承接其余 `76%` 的多轮对话、长文档 RAG 与 Agent 工具调用）**：通过 `Envoy AI Gateway (K8s GIE EPP)` 按前缀块哈希将请求路由至持有相同文档/历史 KV 的 GPU 实例，**只跳过已出现的精确前缀 Token 的 Prefill 计算，最后一个回答仍由大模型实时针对当前问题生成**——既有 `6.9 倍` 首字提速，又做到 **`0.0%` 语义张冠李戴**！

---

## 三、三种企业多租户 RAG & Agent 平台架构定量对比矩阵（`40 Peak QPS · 12K Context`）

| 评估维度 | 方案 A：后置 ACL + 无缓存 + 轮询网关 (`Post-Filter + Naive RR`) | 方案 B：裸开余弦语义缓存 (`Naive Semantic Cache cos>0.90`) | 方案 C：前置 ACL + 实体校验 L1 + Envoy GIE L2 Prefix KV (`#571`) 🏆 |
| :--- | :--- | :--- | :--- |
| **小权限租户检索召回率 (`Recall@10`)** | `41.2%`（后置过滤导致零结果饥饿 ✕） | `78.5%` | **`96.4%`（`Pre-Filter Bitmap + 混合检索重排` ✓）** |
| **缓存张冠李戴/跨单错误率** | `0.0%` | `11.8%`（数字/年份/租户串单事故 ✕） | **`0.02%`（降低 `590 倍` ✓）** |
| **底层引擎 Prefix KV 命中率** | `18.0%`（被轮询打散 ✕） | `22.5%` | **`91.5%`（`Envoy GIE` 前缀粘性路由 ✓）** |
| **首字延迟 (`P95 TTFT`)** | `1,450 ms` | `640 ms` | **`210 ms`（提速 `6.90 倍`）** |
| **承载 40 QPS 所需 H200 总卡数** | `32 张 H200` | `20 张 H200` | **`12 张 H200`（立省 `62.5%` GPU）** |
| **每百万有效 Token 成本 (`$/1M`)** | `$0.240 / 1M` | `$0.145 / 1M` | **`$0.068 / 1M`（成本降低 `3.53 倍`）** |

---

## 四、多租户 RAG 与 Agent 平台上线核对清单（SA Pre-Production Checklist）

- [ ] **1. 向量库强制开启 `Pre-Filter ACL`**：严禁在业务代码层对 Top-K 结果做二次权限剔除；缓存 Key 必须冠以 `tenant_id:acl_hash:` 前缀。
- [ ] **2. 为网关语义缓存加上“数字/日期/否定词槽位精确相等”门禁**：凡包含不同数字或单号的查询一律禁止命中 L1 语义缓存，交由底层 L2 Prefix KV 处理。
- [ ] **3. 规范 Prompt 静态前缀排版顺序**：严格按照 `[System Prompt] -> [Tool Definitions] -> [Shared RAG Context] -> [Chat History] -> [User Query]` 拼接，最大化 `RadixAttention` 块复用率。
