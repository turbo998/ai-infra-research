# 企业级海量离线文档批处理与知识抽取吞吐架构选型指南：同步接口并发死循环（`asyncio.gather` 反模式）vs. 云端半价异步 `Batch API`（`Azure OpenAI / Bedrock Batch`）vs. 自建 `Ray Data + vLLM` 离线引擎 vs. 夜间闲置 `PTU` 零成本回填 + 前缀排序 `Spot GPU Ray/vLLM` + 云端 `Batch API` 混合架构

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-offline-batch-llm-processing-architecture-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、数据工程负责人（Data Platform Lead）与 FinOps 团队的“每晚千万级数据/文档离线大模型批处理、闲置 PTU 削峰填谷与吞吐翻倍”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_offline_batch_llm_processing_architecture_selection/results.json`](../../experiments/enterprise_offline_batch_llm_processing_architecture_selection/results.json)。

![企业级海量离线文档批处理与知识抽取吞吐架构选型总结图](../../assets/svg/enterprise-offline-batch-llm-processing-architecture-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

除实时对话 Copilot 外，企业里有大量 **非实时、高吞吐的离线批处理任务（Nightly Batch ETL）**——例如**每晚在 8 小时窗口内处理 `1,000 万条（10M Records/晚，月均 600 亿 Token = 60B Tokens/月）` 客服录音质检、全量商品打标、财报合同知识图谱抽取或合规审计**。架构师在设计离线批处理管道时最常犯三大致命错误：
1. **反模式一：写个 Python `asyncio.gather()` 脚本直接狂刷线上实时同步 API（Tier 1，每百万 Token 成本高达 `$3.80`，且把线上实时 Copilot 冲垮 `38.5%`）**：数据团队半夜或凌晨启动批处理脚本，直接打在和白天在线客服共用的同步接口上，瞬间触发云端 `HTTP 429` 限流风暴，导致跨时区在线用户大面积报错；而且为明明可以等 8 小时的离线任务支付了最昂贵的实时同步单价，月烧 **`$228,000 / 月`**！
2. **盲区二：放着已经包月付过钱、夜间 85% 闲置的 `PTU`（预留吞吐）不用，额外花全价去买离线算力（Tier 2 / Tier 3）**：企业为了白天高峰已经包月买了 200 PTU，到了晚上 20:00 到次日 07:00 几乎空转——如果不把夜间闲置 PTU 榨干，等于每月白白扔掉近十万美元的免费批处理算力！
3. **盲区三：自建 `vLLM` 跑批处理时按数据库随机主键顺序乱喂数据（Tier 3，GPU 前缀 KV 缓存命中率 `< 8%`）**：1,000 万条数据明明共享着 5,000 份系统模板与长背景文档，因为没做排序，GPU 显存里的 KV Cache 每条都在反复驱逐重算，白白浪费了 **3 倍以上** 的吞吐潜力！

在 **月均 600 亿（60B）批处理 Token、每晚 1,000 万条记录、要求 8 小时内跑完** 的企业基准下，四档离线批处理架构的量化对决如下：

| 方案档位 | 核心离线批处理架构组合 | 每百万 Token 综合单价 | 对线上实时 Copilot 的 `429` 冲击率 | 1,000 万条夜间批处理完结耗时 | 24×7 预留算力综合利用率 | 600 亿 Token 月度总成本 (`TCO`) | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：同步 API 并发死循环 (`Sync asyncio.gather`)** | 直接用高并发协程刷线上实时同步 PAYGO/PTU 接口 | `$3.80 / 1M` ✕ | `38.5%` ✕ *(挤爆在线TPM配额)* | `14.5 小时` ✕ *(超时未跑完)* | `42.0%` | `$228,000 / 月` ✕ | ⛔ 头号工程反模式，既贵又冲垮在线业务 |
| **🟨 Tier 2：100% 云端半价异步 `Batch API` (`Azure / Bedrock Batch`)** | 上传 JSONL 文件走云厂商 24h 异步 Batch API（享 5 折优惠） | `$1.90 / 1M` | `0.0%` *(独立配额池)* | `9.2 小时` | `65.0%` *(未利用闲置PTU)* | `$114,000 / 月` | ✔️ 零运维、天然隔离，适合闭源大模型批处理 |
| **🟦 Tier 3：自建按需 GPU 跑无序 `Ray Data + vLLM`** | 按需购买 H100/L40S 跑离线 vLLM，但按随机顺序输入 | `$1.25 / 1M` | `0.0%` | `5.8 小时` | `74.0%` | `$75,000 / 月` | ⚠️ 成本下降，但未利用前缀排序与闲置 PTU |
| **🟩 Tier 4：夜间闲置 `PTU` 零成本回填 + 前缀排序 `Spot Ray/vLLM` + 云 `Batch API`** | **优先回填已付费夜间闲置 `PTU` (`$0` 边际成本) + 按前缀字典序重排后跑 `Spot GPU Ray Data + vLLM` (`3.1x` 提速) + 云 `Batch API` 兜底** | **`$0.58 / 1M` 🏆** *(-84.7%)* | **`0.0%` 🏆** *(P0/P2优先级秒级让路)* | **`3.4 小时` 🏆** *(快4.3倍)* | **`94.8%` 🏆** *(榨干每分算力)* | **`$34,800 / 月` 🏆** *(-84.7%)* | 🌟 **企业级海量离线批处理终极吞吐与 FinOps 范式** |

> 💡 **架构师一句话选型铁律**：**“设计企业级海量离线大模型批处理，必须用好三把手术刀：① 严禁用 `asyncio.gather` 刷实时接口——闭源前沿模型一律走云端 **5 折异步 `Batch API`**（独立配额池，零 429 冲突）；② 如果企业买了包月 `PTU` 或常驻 GPU，夜间用 **低优先级（P2）削峰填谷控制器** 免费吃掉 20:00–07:00 的闲置算力；③ 自建 `Ray Data + vLLM` 跑海量批处理前，**先对输入表按 `(模板ID, 共享长文档ID)` 做一次字典序排序（Lexicographical Prefix Sort）**——让共享同一份长背景的请求排在一起进 GPU，KV Cache 命中率从 8% 暴涨到 `94%`，吞吐直接翻 `3.1 倍`！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大配额打架、夜间 PTU 空转与离线 GPU 吞吐翻倍痛点）

### Q1：为什么用 `Azure OpenAI Global Batch / AWS Bedrock Batch API` 不仅价格直接打 5 折，而且绝对不会影响白天的线上业务？那什么时候又该上自建 `Ray Data + vLLM`？
- **大白话与第一性原理**：
  1. **云端异步 `Batch API`（Tier 2）的机制**：你只需把 100 万条请求打包成 `.jsonl` 文件上传到云存储，云厂商利用其全球数据中心的夜间空闲算力在 24 小时内异步跑完并输出 `.jsonl` 结果。它有两个巨大红利：**第一，单价直接打 5 折（`-50%`）；第二，它使用完全独立的后台配额池，哪怕一次提交 1,000 万条也绝不会占用你线上实时接口的半点 TPM/RPM 配额（`0.0%` 429 冲突）**！
  2. **什么时候选云端 `Batch API`，什么时候选 `Spot GPU + Ray Data + vLLM`？**：
     - 如果任务必须调用闭源前沿大模型（如 `GPT-4o / Claude 3.5 Sonnet`），或者批处理是偶发性的（月均 `< 100 亿 Token`）👉 **直接用云端 5 折 `Batch API`**；
     - 如果任务是固定模式的分类、抽取、翻译或打标（可用开源/蒸馏 `8B~70B` 模型），且要求**必须在每天凌晨 4~6 小时窗口内准时跑完入仓** 👉 **用 `Spot GPU (L40S / H100) + Ray Data + vLLM Offline`**，综合单价可进一步压低至 **`$0.58 / 百万 Token`**！

### Q2：怎么安全地把白天包月买的 `PTU` 夜间闲置算力（20:00–07:00）白嫖用来跑离线批处理，同时保证万一半夜有海外在线用户访问时 `0.0%` 卡顿？
- **网关级 `P0 / P2` 优先级令牌桶回填控制器（Priority-Aware Idle PTU Scavenger）**：
  - 在 AI 网关前部署双队列：在线实时 Copilot 标记为 **`P0（最高优）`**，离线批处理队列标记为 **`P2（填充级）`**；
  - 控制器每 5 秒读取一次当前 `PTU` 实时利用率 $U_{\text{online}}$，并按 $\text{Rate}_{\text{batch}} = \max(0, 0.85 - U_{\text{online}}) \times \text{TPM}_{\text{PTU}}$ 的速率向 PTU 端点平滑注入离线批处理任务；
  - 一旦监测到在线 `P0` 请求上升或 PTU 利用率超过 75%，网关在 **`< 100ms` 内立即暂停 `P2` 批处理注入**，把 100% 通道让给在线用户——不花一分钱额外算力费，就把全月 **45% 的离线批处理任务** 在夜间闲置 PTU 上免费跑完了！

### Q3：为什么在把 1,000 万条数据喂给离线 `vLLM` 之前，仅仅在 Spark/Ray 里加一行 `ORDER BY (template_id, doc_id)`，就能让 GPU 批处理速度暴涨 `3.1 倍`？
- **字典序前缀局部性原理（Lexicographical Prefix Locality for GPU KV Cache）**：
  - 假设你有 1,000 万条商品评论或合同条款需要抽取，它们分别引用了 5,000 份长达 6,000 Token 的品类规范或主合同全文。
  - **如果不排序（Tier 3）**：第 1 条处理合同 A，第 2 条处理合同 B……到了第 500 条又遇到合同 A 时，合同 A 的 6,000 Token KV Cache 早就从 GPU 显存里被挤出去了，GPU 只能痛苦地一遍遍重复做 6,000 Token 的 Prefill 计算（KV 命中率 `< 8%`）！
  - **如果先按 `(template_id, doc_id)` 排序（Tier 4）**：所有引用合同 A 的 2,000 个子问题会**紧挨着在同一秒进入同一张 GPU 的 Continuous Batch**！第一条算完 6,000 Token 的合同背景后，紧随其后的 1,999 条全部 **`100%` 命中显存里的 Prefix KV Cache**，平均完结耗时从 `5.8 小时` 直接缩短至 **`3.4 小时`（提速 `3.1x`）**！

---

## ③ 🧭 直观可视化图解（选型决策树 + 三通道混合批处理架构图 + 削峰填谷与前缀排序时序图）

### 3.1 企业级海量离线文档与数据批处理架构选型决策树

```mermaid
flowchart TD
    A["企业级海量离线批处理架构选型<br/>(月均 600 亿 Token + 每晚 1,000 万条记录)"] --> B{"是否还在用 asyncio.gather 并发调用线上实时同步 API?"}
    B -- "是 (Tier 1 反模式)" --> C["⛔ 立即叫停! 既付全价 ($228K/月)<br/>又引发 HTTP 429 风暴冲垮线上实时 Copilot (38.5% 降级)"]
    B -- "否 (走专用异步批处理架构)" --> D{"企业是否拥有白天包月预留的 PTU 或常驻 GPU 集群?"}
    D -- "有 (夜间 20:00-07:00 存在大量闲置波谷)" --> E["♻️ 第一顺位：启用 P2 优先级闲置 PTU 削峰填谷器<br/>以 $0 边际成本免费消化 40%~50% 的夜间批处理量!"]
    D -- "无闲置 PTU 或超出夜间 PTU 剩余容量" --> F{"批处理模型类型与完成时间窗口要求?"}
    E --> F
    F -- "闭源前沿大模型 (GPT-4o) 且允许 24h 内完成" --> G["🟨 云托管 5 折异步 Batch API (Azure Global Batch / Bedrock Batch)<br/>(独立配额池 | 零运维 | 成本立减 50%)"]
    F -- "开源/蒸馏模型 (8B~70B) 且要求 <4h 极速入仓" --> H["🟩 Tier 4: 前缀字典序排序 + Spot GPU (Ray Data + vLLM) 🏆<br/>(KV 命中率 94% | 吞吐翻 3.1 倍 | $0.58/百万 Token | 省 84.7% TCO)"]
    style E fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style H fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 夜间闲置 `PTU` 回填 + 前缀排序 `Spot Ray/vLLM` + 云 `Batch API` 参考架构图

```mermaid
flowchart LR
    subgraph Prep["1. 前缀局部性排序与分片层"]
        Raw["📦 1,000 万条夜间待处理记录<br/>(Parquet / Iceberg / Delta)"] --> Sort["🗂️ 字典序前缀排序器<br/>ORDER BY (template_id, shared_doc_id)<br/>把相同长背景的请求聚簇在一起"]
    end

    subgraph Channels["2. 三级低成本高吞吐执行通道"]
        Sort -->|通道 A: 45% 流量 ($0 边际成本)| PTU["♻️ 夜间闲置 PTU 削峰填谷器 (P2 优先级)<br/>仅当在线 P0 利用率 < 70% 时注入<br/>在线流量一来 <100ms 立即让路"]
        Sort -->|通道 B: 45% 流量 (3.1x 极速吞吐)| Ray["⚡ Spot GPU (Ray Data + vLLM Offline)<br/>因输入已按前缀排序，KV 缓存命中率达 94%!<br/>每 1 万条自动写 Parquet 断点续传"]
        Sort -->|通道 C: 10% 闭源难例溢出| BatchAPI["☁️ 云端 5 折异步 Batch API<br/>(Azure OpenAI / Bedrock Batch)"]
    end
```

### 3.3 从“前缀字典序重排”到“夜间闲置 PTU 零成本回填与 Spot 断点续传”时序图

```mermaid
sequenceDiagram
    autonumber
    participant ETL as 夜间数仓调度器 (20:00 启动)
    participant Sorter as Ray Data 前缀排序与分片器
    participant Scav as 闲置 PTU 削峰填谷控制器
    participant Spot as Spot GPU vLLM 离线池
    participant Out as 目标数仓表 (Parquet Checkpoints)

    ETL->>Sorter: 提交 1,000 万条待抽取记录
    Sorter->>Sorter: 按 `(system_template_id, parent_doc_id)` 字典序排序并切分为 1,000 个微批次（每批 1 万条）
    par 双擎并发消化（3.4 小时全部跑完）
        Sorter->>Scav: 分派前 450 个微批次至夜间闲置 PTU 回填通道
        Scav->>Out: 在不触碰在线 P0 流量水位的前提下，以 $0 边际成本完成 450 万条推理！
        Sorter->>Spot: 分派后 550 个微批次至 Spot GPU Ray Data + vLLM 池
        Spot->>Out: 因同文档请求连续进入同一 Batch，GPU KV 命中率达 94%（吞吐提升 3.1 倍），每完结 1 万条落盘一次 Parquet（不怕 Spot 抢占）！
    end
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：同步 API `asyncio.gather` | 🟨 Tier 2：100% 云端 5 折 `Batch API` | 🟦 Tier 3：按需 GPU 无序 `Ray + vLLM` | 🟩 Tier 4：闲置 `PTU` 回填 + 前缀排序 `Spot Ray/vLLM` |
|---|---|---|---|---|
| **每百万 Token 综合单价** | `$3.80 / 1M` ✕（付全价实时费） | `$1.90 / 1M`（官方 5 折） | `$1.25 / 1M` | 🏆 **`$0.58 / 1M`（45% 免费吃闲置 PTU + Spot GPU）** |
| **对线上实时 Copilot 的冲击** | `38.5%` ✕（引发 429 限流雪崩） | `0.0%`（云端独立配额池） | `0.0%` | 🏆 **`0.0%`（`P0/P2` 优先级隔离，`<100ms` 秒级让路）** |
| **1,000 万条夜间批处理耗时** | `14.5 小时` ✕（白天还没跑完） | `9.2 小时`（受 24h 队列调度影响） | `5.8 小时`（KV 频繁失效） | 🏆 **`3.4 小时`（字典序前缀排序令 KV 命中率达 94%）** |
| **24×7 预留算力综合利用率** | `42.0%` ✕ | `65.0%`（夜间 PTU 空转） | `74.0%` | 🏆 **`94.8%`（白天保在线 SLA，夜间跑满离线批处理）** |
| **600 亿 Token/月总成本 (`TCO`)** | `$228,000 / 月` | `$114,000 / 月` | `$75,000 / 月` | **`$34,800 / 月` 🏆（节省 84.7%，年省 $231 万）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（月均 600 亿批处理 Token / 每晚 1,000 万条记录 / 8 小时时间窗）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级海量离线文档与数据批处理大模型定容计算器](../reports/enterprise-offline-batch-llm-processing-architecture-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-offline-batch-llm-processing-architecture-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_offline_batch_llm_processing_architecture_selection/run.py --check
  python -m unittest tests/test_enterprise_offline_batch_llm_processing_architecture_selection.py
  ```

### 5.2 核心量化对比图（24×7 算力利用率 % vs. 线上实时业务受冲击降级率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Sync asyncio.gather Loop", "metric": "24x7 Fleet Utilization (%)", "pct": 42.0 },
      { "tier": "Tier 1: Sync asyncio.gather Loop", "metric": "Online Copilot SLA Drop (%)", "pct": 38.5 },
      { "tier": "Tier 2: 100% Cloud Batch API", "metric": "24x7 Fleet Utilization (%)", "pct": 65.0 },
      { "tier": "Tier 2: 100% Cloud Batch API", "metric": "Online Copilot SLA Drop (%)", "pct": 0.0 },
      { "tier": "Tier 3: Unsorted On-Demand Ray+vLLM", "metric": "24x7 Fleet Utilization (%)", "pct": 74.0 },
      { "tier": "Tier 3: Unsorted On-Demand Ray+vLLM", "metric": "Online Copilot SLA Drop (%)", "pct": 0.0 },
      { "tier": "Tier 4: Idle PTU + Sorted Spot Ray/vLLM", "metric": "24x7 Fleet Utilization (%)", "pct": 94.8 },
      { "tier": "Tier 4: Idle PTU + Sorted Spot Ray/vLLM", "metric": "Online Copilot SLA Drop (%)", "pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 boosts 24x7 fleet utilization to 94.8% with 0.0% online SLA impact",
    "subtitle": "Comparison across 60 Billion monthly batch tokens and 10M nightly records",
    "encodings": {
      "x": "tier",
      "y": "pct",
      "group": "metric",
      "color": "metric"
    }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地核验清单（Production Readiness Checklist）

- [ ] **明令禁止数据工程团队使用 `asyncio.gather()` 直接对生产实时同步接口跑海量离线批处理**：物理隔离在线 `P0` 与离线 `P2` 流量通道。
- [ ] **在 AI 网关层部署“夜间闲置 `PTU` / 常驻 GPU 削峰填谷回填控制器”**：当实时在线利用率 `< 70%` 时自动注入离线队列，一旦在线流量上升在 `< 100ms` 内无条件暂停离线任务。
- [ ] **在启动 `Ray Data + vLLM` 离线批处理前，强制对输入数据集执行 `ORDER BY (system_template_id, shared_context_doc_id)` 字典序排序**：最大化 GPU 显存内的 Prefix KV Cache 复用率（将吞吐提升 `2.5x~3.1x`）。
- [ ] **为运行在 Spot GPU 上的批处理作业配置微批次断点续传（每 `10,000 条记录` 落盘一次 Parquet 分片）**：确保 Spot 节点被云厂商回收时仅损失几十秒进度而非从头重跑。
- [ ] **对闭源前沿大模型（`GPT-4o / Claude`）的非紧急批处理任务，默认路由至云厂商 5 折异步 `Batch API`（`Azure OpenAI Global Batch / AWS Bedrock Batch`）**。
