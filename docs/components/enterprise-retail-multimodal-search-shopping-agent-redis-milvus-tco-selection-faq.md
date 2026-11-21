# 企业级电商多模态商品搜索、视觉以图搜图与导购 Agent 选型：`Azure AI Search + SigLIP` vs. `Vertex AI Search for Retail / Algolia` vs. `自建 Milvus 62ms 快慢双轨 + Redis 实时库存锚定` 转化率与 TCO 指南

> **语言 / Language**: **简体中文** | [English](./enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-faq.en.md)  
> **配套交互式学习与精算工具**: [📱 交互式 H5 精读页 (电商多模态搜索快慢双轨与转化率沙盘)](../h5/retail-multimodal-search-shopping-agent-milvus-redis-lab.html) · [🧮 HTML 交互式计算器](../reports/enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-sizer.html) · [🃏 7-Card 4:5 视觉评测卡](../reports/enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-cards.html) · [📊 可复核模型](../../experiments/enterprise_retail_multimodal_search_shopping_agent_redis_milvus_tco_selection/) · [🎓 返回 AI Infra 全景精学馆](../h5/index.html)

![企业级电商多模态商品搜索与导购 Agent 62ms 快慢双轨架构图](../../assets/svg/enterprise-retail-multimodal-search-shopping-agent-redis-milvus-tco-selection-faq-summary.svg)

---

## 一、3 分钟看懂核心结论（Executive Summary）

> **一句话决策结论**：电商与零售搜索框存在一条铁血生死线：**首屏商品瀑布流渲染 `P99 <= 100ms`（延迟每增加 `100ms`，GMV 下单转化率直接下跌 `1.2% ~ 1.8%`）**！在改造传统电商搜索与上线 AI 导购时，**最致命的两大架构事故是：① 在主搜索链路上同步阻塞调用大模型做 Query 改写（把首屏延迟拖慢到 `1,680ms`，用户未等结果出来已关掉 App，转化率反而暴跌 `-18.5%`）；② 把商品的促销价格与尺码库存写进向量数据库的文本 Chunk 里（导致导购 Agent 拿着昨天的过期价格或已断码售罄的商品向用户推荐，误推率高达 `14.8%`）**！
> - **追求开箱即用的云原生零售团队**：推荐 **`Azure AI Search (多模态向量 + BM25 + Semantic Ranker)`**（首屏 `74ms`，月成本 `$29,500`）或 **`Google Vertex AI Search for Retail / Algolia`**（内置电商 CTR/CVR 归因模型）。
> - **5,000 万+ SKU、高并发大促（`5,000+ QPS`）的头部零售平台**：必须采用 **`Tier 4 离线 VLM 商品图深度打标 + 在线 62ms 快慢双速解耦（Milvus SigLIP 多模态混合检索 + GBDT 商业精排）+ 1.5ms Redis 毫秒级库存价格锚定 + 异步 SSE 导购卡片`**：
>   1. **把大模型的重推理 100% 挪到离线入库期**：夜间用 `Qwen2.5-VL-7B` 扫描商家上传的商品主图，自动补全“法式方领、茶歇风、海边度假、梨形身材显瘦”等结构化视觉标签写入倒排与向量索引；
>   2. **在线 `62ms` 快慢双轨分流**：快轨（Fast Track）零大模型阻塞，`62ms` 秒出图文混合召回的商品列表；慢轨（Slow Track）以异步 SSE 流式渲染顶部穿搭导购卡；
>   3. **`1.5ms` Redis 真值覆盖**：严禁从向量库读价格！召回 `Top-50 SKU_ID` 后花 `1.5ms` 从 Redis 批量 `MGET` 实时库存与到手价，**缺货/错价率 `0.0%`，GMV 下单转化率净增 `+21.8%`，月成本从 `$16.4 万` 暴降至 `$16,200/月`（省 `90.1%`）**！

| 核心选型维度 | `Tier 1: 同步阻塞调 70B 改写搜索 + 向量库静态价格` | `Tier 2: 垂直零售搜索 SaaS (Vertex Retail / Algolia)` | `Tier 3: 云原生多模态检索 (Azure AI Search + SigLIP)` | `Tier 4: 离线 VLM 增广 + 62ms 快慢双轨 + Redis 库存锚定 (🏆)` |
|---|---|---|---|---|
| **首屏商品瀑布流延迟 (`Search P99`)** | `1,680 ms` ✕ (白屏转圈跳失) | `85 ms` ✓ | `74 ms` ✓ | **`62 ms` (零大模型阻塞主搜索)** 🏆 |
| **推荐已售罄/过期价格事故率** | `14.8%` ✕ (向量库T+1延迟) | `0.4%` | `0.1%` ✓ | **`0.0%` (`1.5ms` Redis `MGET` 实时过滤)** 🏆 |
| **以图搜图与模糊场景词召回率** | `78.5%` | `89.4%` ✓ | `92.6%` ✓ | **`96.2%` (离线 VLM 视觉打标 + SigLIP)** 🏆 |
| **GMV 下单转化率净提升 (`CVR Lift`)** | **`-18.5%` ✕ (因卡顿反跌!)** | `+15.2%` ✓ | `+16.5%` ✓ | **`+21.8%` (秒开列表 + 异步导购卡)** 🏆 |
| **5,000 万 SKU 月度总成本 (`TCO`)** | `$164,000 / 月` ✕ | `$58,000 / 月` | `$29,500 / 月` ✓ | **`$16,200 / 月` (-90.1%)** 🏆 |

---

## 二、核心技术原理与架构师灵魂拷问（Q&A）

### Q1：为什么商家上传商品时只写了“夏季连衣裙”五个字，用户搜“适合去三亚拍照的显瘦茶歇裙”却能精准秒级召回？“离线 VLM 增广”是怎么把大模型延迟从在线抹掉的？
1. **商家的“懒惰标题”痛点**：平台上 80% 的中小商家上架商品时，标题只会堆砌“2026新款女装夏季连衣裙”，根本不会写领型、袖长、面料垂感、适用身材或穿搭场景。
2. **错误做法（在线猜词） vs. 正确做法（离线看图扩写）**：
   - 如果等用户在搜索框输入“三亚拍照显瘦茶歇裙”时，在线调用大模型去理解改写，每次搜索都要多等 `800ms` 且每秒消耗 5,000 次 LLM 调用！
   - **离线 VLM 增广（Offline Multimodal Catalog Enrichment）**：在商品上架入库的离线流水线中（利用夜间闲时 GPU 跑 `Qwen2.5-VL-7B`，每张商品图成本不足 `$0.0002`），让视觉大模型仔细观察商品主图与尺码表，自动生成包含 `30+` 个维度的结构化 JSON 属性（`{"neckline": "方领", "style": "法式茶歇", "occasion": ["海边度假", "三亚拍照"], "body_fit": "高腰遮胯/梨形友好"}`），同时用 **`SigLIP-SO400M`** 提取图文对齐向量；
   - 当用户在线搜索或上传一张小红书穿搭图片（以图搜同款）时，在线引擎只需做 **`18ms` 的 `BM25 + SigLIP 向量` 双路召回**，无需在线调任何大模型即可精准命中！

```mermaid
flowchart TB
    subgraph Offline["🌙 离线商品入库增广流水线 (夜间闲时 GPU · $0.0002/SKU)"]
        Img["商家上传商品主图 + 简陋标题"] --> VLM["Qwen2.5-VL-7B 视觉属性抽取<br/>自动打标：领型/材质/穿搭场景/适用身材"]
        VLM --> SigLIP["SigLIP-SO400M 图文统一编码<br/>写入 Milvus / Azure AI Search 混合索引"]
    end

    subgraph Online["⚡ 在线 62ms 快慢双速搜索与导购流水线 (5,000 QPS)"]
        User["👤 用户输入文字或拍照搜同款<br/>'适合去海边拍照的显瘦方领裙'"] --> Fast["🏎️ 快轨 (Fast Track · 62ms 首屏渲染)<br/>1. BM25 + SigLIP 向量召回 Top-200 (18ms)<br/>2. Redis MGET 实时库存/到手价，剔除断码品 (1.5ms)<br/>3. GBDT 商业 CTR/CVR 精排输出商品流 (22ms)"]
        User --> Slow["🧠 慢轨 (Slow Track · 异步 SSE 导购卡)<br/>不阻塞下方商品列表！基于快轨 Top-6 真实有货 SKU<br/>流式输出尺码建议与防晒开衫搭配推荐"]
    end
```

---

## 三、生产落地 Checklist（Pre-Production Checklist）

- [ ] **主搜索链路严禁同步阻塞调用生成式 LLM**：首屏商品列表必须由 `倒排 + 向量 + 传统 GBDT 精排` 在 `<100ms` 内直接返回，AI 导购总结通过异步 SSE 独立渲染。
- [ ] **向量数据库严禁存储实时价格、促销券后价与尺码库存数**：向量库只存 `SKU_ID` 与静态商品特征，召回后必须经 `Redis MGET`（`<2ms`）实时注入最新价格并过滤 `stock == 0` 的断码商品。
- [ ] **融合商业转化率信号（`LTR / GBDT Re-ranking`）**：纯语义相似度（Cosine Similarity）会把“描述很像但差评如潮、零销量的滞销品”排在第一名；必须在语义召回分之上叠加实时点击率（`CTR`）、转化率（`CVR`）与店铺评分精排。

---

## 四、来源与方法论溯源（Provenance）

- **数据来源声明**：本指南实验数据来自可复核解析模型 [`experiments/enterprise_retail_multimodal_search_shopping_agent_redis_milvus_tco_selection/run.py`](../../experiments/enterprise_retail_multimodal_search_shopping_agent_redis_milvus_tco_selection/run.py)（`Provenance: synthetic + anonymized_reported, independently_reproduced=false`）。
