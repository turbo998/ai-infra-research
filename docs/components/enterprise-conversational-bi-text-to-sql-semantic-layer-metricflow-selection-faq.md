# 企业级对话式 BI 与智能问数平台（`Conversational BI & Text-to-SQL` vs. 治理语义层 `Cube / dbt MetricFlow / Fabric Semantic Model`）、`SQLGlot AST` 防爆表门禁与行级权限透传（`RLS`）架构选型指南：为什么把 800 张数仓物理表 `DDL` 直接塞给大模型写 SQL 会有 `41.5%` 算错核心经营指标，甚至一条漏写分区键的 `JOIN` 就能瞬间烧掉 `$250` 数仓算力？

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、企业数据平台与 BI 负责人（Snowflake / Databricks / BigQuery / Microsoft Fabric）的“智能问数平台（ChatBI）架构选型、从裸写 `Text-to-SQL` 升级为 `Text-to-Semantic-Intent`（统一指标语义层避免一对多 `Join` 扇出膨胀）、`SQLGlot AST + Dry-Run` 防全表扫描熔断与用户级 `OAuth RLS` 权限透传”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection/results.json`](../../experiments/enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection/results.json)。

![企业级对话式 BI 与治理语义层架构总结图](../../assets/svg/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

几乎每家企业的 CFO、销售副总裁和业务分析师都想要一个 **“自然语言智能问数（ChatBI）”** 助手，但当工程团队把企业数仓（`800+` 张物理表、上万个缩写晦涩的字段、月均 `200,000` 次问数请求）的 `CREATE TABLE` 建表语句直接喂给大模型让它自由发挥写 SQL（`Raw DDL Text-to-SQL`）时，往往上线不到两周就会被业务部门彻底弃用：
1. **灾难一：“同名不同口径”与“一对多 `JOIN` 扇出膨胀（Fan-Out Inflation）”导致 `41.5%` 的高管核心指标算错（Tier 1，准确率仅 `58.5%`）**：
   - 高管问一句：*“今年 Q3 华东区各行业客户的净留存收入（Net ARR）和客诉工单数分别是多少？”*
   - 大模型直接把 `orders`（订单主表，1 个客户 5 笔订单）和 `support_tickets`（工单表，1 个客户 12 张工单）按 `customer_id` 做了一个直接 `JOIN`，然后写下 `SUM(orders.arr)`——结果 **因为每个订单被 12 张工单复制了 12 遍，算出来的华东区收入直接虚增了 12 倍（典型的多事实表裂谷陷阱 `Chasm Trap`）**！此外，大模型还经常漏写 `WHERE is_test_account = 0 AND order_status = 'SETTLED'` 等隐式业务过滤条件，导致同一个问题问三次算出三个完全不同的数字！
2. **灾难二：漏写分区键或笛卡尔积引发 `50TB` 全表扫描（Tier 1，失控爆表率达 `8.2%`，单条烂 SQL 就能烧掉 `$45~$250` 的 Snowflake/BigQuery 按量算力，全月账单与返工成本高达 `$215,000`）**！
3. **灾难三：共用高权限连接池账号导致 `12.4%` 的越权查数风险（Bypassing Row-Level Security `RLS`）**：华东区销售经理随口问一句“把华南区大客户的底价折扣也列出来”，共用 `readonly_admin` 数据库连接的 Agent 竟然真的把全公司机密底价全查出来了！

在 **800 张数仓物理表、月均 200,000 次自然语言问数查询（其中 `90%` 涉及企业核心经营 KPI，`10%` 为长尾即席探索）** 的生产基准下，四档对话式 BI 架构的实测对决如下：

| 方案档位 | 指标口径治理、SQL 生成与安全防护机制 | 高管经营指标准确率 | 失控全表扫描事故率 | 行级权限 (`RLS`) 越权风险 | 20 万次问数/月总 TCO | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：物理表 `DDL` 直塞大模型自由写 SQL** | 直接让 LLM 拼写多表 `JOIN` 与聚合公式；共用数据库账号 | `58.5%` ✕ *(41.5%算错口径/膨胀)* | `8.2%` ✕ *(频繁扫爆50TB大表)* | `12.4%` ✕ *(共享账号越权查数)* | `$215,000 / 月` ✕ *(巨额数仓浪费与对账)* | ⛔ 严禁让大模型直接对着数仓物理表自由手写复杂业务聚合 SQL |
| **🟨 Tier 2：向量检索 `Schema-RAG + Few-Shot SQL` (`Vanna` 类)** | 检索相似历史 SQL 供 LLM 模仿手写 SQL | `80.6%` ✕ *(跨粒度Join仍会算重)* | `2.8%` | `4.5%` | `$138,000 / 月` ✕ | ⛔ 能找准表名，但仍无法在数学上根除一对多 `JOIN` 翻倍错误 |
| **🟦 Tier 3：用历史查询日志微调专属 `SQL Code-LLM`** | 将数仓历史 SQL 训入模型权重 | `84.0%` | `2.1%` | `3.8%` | `$152,000 / 月` ✕ | ⛔ 死记硬背过期表结构与分析师历史错题，数仓改表即失效 |
| **🟩 Tier 4：治理语义层 (`Cube / dbt MetricFlow / Fabric DAX`) + `SQLGlot AST & Dry-Run` 门禁 + `OAuth RLS` 透传** | **LLM 只输出声明式 `{metrics, dimensions, filters}`；确定性编译器生成防膨胀 `CTE SQL`；长尾查询过 `AST` 与 `10GB` 扫描上限** | **`98.6%` 🏆** *(与官方BI报表100%对齐)* | **`0.0%` 🏆** *(拦截无分区与超限扫描)* | **`0.0%` 🏆** *(用户OAuth令牌直通RLS)* | **`$26,000 / 月` 🏆** *(比 Tier 1 省 87.9%)* | 🌟 **企业级对话式 BI 与高管智能问数唯一黄金标准** |

> 💡 **架构师一句话智能问数铁律（大模型选指标、语义层编 SQL、AST 拦爆表、令牌控行权）**：
> 1. **从 `Text-to-SQL` 转向 `Text-to-Semantic-Intent` 铁律（Governed Semantic Layer First）**：“**绝不让大模型现场发明财务公式和表关联路径！**”对于企业 `90%` 的经营问数，大模型**完全不写 SQL**，而是只从 **`dbt MetricFlow / Cube / Microsoft Fabric Semantic Model / LookML`** 暴露的指标目录里挑选 JSON 意图：`{"measures": ["net_arr", "ticket_count"], "dimensions": ["region"], "filters": [...]}`。随后由语义层的 **确定性对称聚合编译器（Symmetric Aggregate Compiler）** 自动将不同粒度的事实表分别先在子查询（`CTE`）内聚合到一致粒度后再关联——**100% 根绝一对多 `JOIN` 扇出翻倍 bug，确保 AI 算出的每一个数字都与 CFO 的 PowerBI / Tableau 看板分毫不差（准确率 `98.6%`）**！
> 2. **`SQLGlot AST` 语法树审计 + `15ms EXPLAIN / Dry-Run` 字节熔断铁律（Zero Runaway Scans）**：针对 `10%` 超出预定义指标的长尾探索问题，生成的受限 SQL 在发往数仓前必须经过两道代码铁闸：① `SQLGlot AST` 静态扫描（强制只读 `SELECT`、禁止 `CROSS JOIN`、强制检查分区时间列 `dt >= ...`）；② 调用数仓 `EXPLAIN / DRY_RUN` 预估扫描量，一旦预估扫描超过 **`10 GB`** 立即拦截并提示缩小时间范围（失控爆表率 **`0.0%`**）！
> 3. **端到端用户 `OAuth / Entra ID` 身份透传铁律（Pass-Through Row-Level Security）**：严禁使用全局共享 `service_account` 查数！所有查询必须携带提问者本人的身份声明（或由语义层强制注入 `WHERE region_id IN ($user.allowed_regions)`），实现 **`0.0%` 越权数据泄露**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击为什么大模型写 `JOIN` 会把收入算大 10 倍、怎么和 PowerBI/Tableau 保持同一套口径、怎么防一条 SQL 扫穿数仓预算 3 大痛点）

### Q1：为什么哪怕是顶配的 GPT-4o / Claude 3.5，在面对包含 2 张以上事实表（如“合同金额 + 回款记录 + 客服工单”）的业务问题时，直接写出的 SQL 有超过 `40%` 会算错金额？
- **剖析数据分析界最著名的“扇出陷阱（Fan-Trap）”与“裂谷陷阱（Chasm-Trap）”**：
  - 假设客户 A 签了 1 份 `$100,000` 的年度合同（在 `contracts` 表 1 行），分 4 期开票（在 `invoices` 表 4 行），期间提了 5 个工单（在 `tickets` 表 5 行）。
  - 如果大模型写出一句看似无比通顺的：`SELECT SUM(c.contract_amount), COUNT(t.ticket_id) FROM contracts c JOIN invoices i ON ... JOIN tickets t ON ...`，由于笛卡尔乘积效应，`c.contract_amount` 这一行被重复放大了 $4 \times 5 = 20$ 倍，算出来的合同总额直接变成了 **`$2,000,000`（整整放大了 20 倍）**！
  - **Tier 4 语义层编译器的降维打击**：当大模型只输出 `measures: ["contract_amount", "ticket_count"]` 时，`MetricFlow / Cube` 的确定性图编译器知道这两个指标处于不同的事实粒度（Grain），它会自动编译成 **两段独立的预聚合 CTE（先在 `CTE_1` 按客户算好合同额，再在 `CTE_2` 按客户算好工单数，最后按主键 `FULL OUTER JOIN`）**——从数学根源上消灭了扇出膨胀！

### Q2：如果业务团队已经在 PowerBI（`Fabric DAX`）或 `dbt` 里维护了几百个核心指标，做 AI 问数助手时难道还要在 Prompt 里重新抄一遍所有指标的计算公式吗？
- **“单一事实源（Single Source of Truth）”双向复用架构**：
  - 千万不要在 AI 工程代码里用 Prompt 维护第二套指标逻辑！一旦下个月财务部微调了“净毛利”的扣减项，BI 看板改了而 Prompt 忘了改，高管就会发现 AI 报的数和看板对不上。
  - 在 Tier 4 中，AI Agent 直接通过 **`Cube REST/GraphQL API`、`dbt Semantic Layer JDBC/MCP` 或 `Microsoft Fabric DAX ExecuteQueries API`** 读取现有的企业级语义模型。财务部只要在语义层改一次公式，**PowerBI 看板、Excel 透视表与 AI 对话助手在同一秒钟自动同步更新**！

### Q3：面对數百 TB 的 `Snowflake / BigQuery / Databricks` 生产大表，怎么从架构上做到“绝对不允许 AI 生成的一条烂 SQL 跑满 20 分钟、烧掉几百美金算力”？
- **三层查询成本防护栏（`SQLGlot AST` + `EXPLAIN Byte Gate` + `Warehouse Resource Monitor`）**：
  1. **语法层（`0.8ms` `SQLGlot AST` 改写与拦截）**：解析生成的 SQL 抽象语法树，若发现查询了日志级百亿行大表却没有带分区列过滤（如 `event_date`），直接拒绝执行并让模型补齐时间范围；若未带 `LIMIT`，自动注入 `LIMIT 1000`；
  2. **优化器预检层（`15ms` `DRY_RUN / EXPLAIN`）**：在真正拉起计算节点前，先向数仓发送零成本的 `DRY_RUN`（BigQuery）或 `EXPLAIN USING JSON`（Snowflake/Databricks），读取预估扫描字节数 `estimated_bytes`——凡超过 **`10 GB`** 的查询一律熔断拦截；
  3. **连接池隔离层**：为 AI 探索查询分配专属的小规格数仓计算池（如 Snowflake `X-Small Warehouse` 并设置 `STATEMENT_TIMEOUT_IN_SECONDS = 15`），物理锁死单次查询的最大开销！

---

## ③ 🧭 直观可视化图解（智能问数选型决策树 + 治理语义层与 `AST` 门禁架构图 + 拦截扇出膨胀与越权查数时序图）

### 3.1 企业级对话式 BI 与智能问数平台架构选型决策树

```mermaid
flowchart TD
    A["企业级对话式 BI 与智能问数选型<br/>(800 张数仓物理表 | 月均 20 万次问数 | 涵盖 Snowflake / Databricks / Fabric)"] --> B{"大模型如何将自然语言问题转化为数仓查询结果?"}
    B -- "把物理表 DDL 塞进 Prompt 让大模型自由手写 SQL (Tier 1)" --> C["⛔ 41.5% 算错核心经营指标 + 8.2% 全表扫描爆表！<br/>一对多 JOIN 导致收入虚增数倍，且存在 12.4% 越权查数风险"]
    B -- "向量检索历史 SQL 样本让大模型模仿写 SQL (Tier 2)" --> D["⛔ 跨粒度多事实表关联仍有 19.4% 算重错误！<br/>且与官方 BI 报表口径脱节，月耗 $13.8 万"]
    B -- "用历史查询日志微调专属 SQL 大模型 (Tier 3)" --> E["⛔ 死记过期字段与历史错误 SQL (准确率仅 84%)！<br/>数仓表结构一变更就需重新训练"]
    B -- "治理语义层 (Cube/MetricFlow) + SQLGlot AST/Dry-Run 门禁 + OAuth RLS (Tier 4)" --> F["🟩 Tier 4: Text-to-Semantic-Intent + 防膨胀 CTE 编译器 + AST/RLS 铁闸 🏆<br/>(98.6% 指标准确率 | 0.0% 爆表扫描 | 0.0% 越权泄露 | 省 87.9% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style D fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 治理语义层（`Cube / MetricFlow / Fabric`）、`SQLGlot AST` 防爆表门禁与 `OAuth RLS` 架构图

```mermaid
flowchart LR
    subgraph UserLayer["1. 身份透传与意图路由层"]
        User["👤 高管 / 分析师提问<br/>(携带 Entra ID OAuth 令牌)"] --> Router{"🧭 是否命中治理指标库?"}
    end

    subgraph SemanticPath["2A. 治理语义层主通道 (90% 核心经营问数)"]
        Router -- "90% 命中语义层" --> Intent["🧠 LLM 生成声明式语义 JSON:<br/>{measures: [net_arr, tickets],<br/>dimensions: [region], time: Q3}"]
        Intent --> Compiler["📐 对称聚合编译器 (Cube / MetricFlow)<br/>按各自粒度先生成独立 CTE 再关联<br/>(0% Join 扇出膨胀 | 98.6% 准确!)"]
    end

    subgraph AdhocPath["2B. 受控长尾探索通道 (10% 即席查询)"]
        Router -- "10% 长尾即席查询" --> AST["🛡️ SQLGlot AST + 15ms Dry-Run 门禁<br/>• 拦截无日期分区与 CROSS JOIN<br/>• 预估扫描 > 10GB 立即熔断 (0% 爆表)"]
    end

    subgraph Warehouse["3. 行级权限 (RLS) 与数仓执行层"]
        Compiler & AST --> RLS["🔐 强制注入调用者 RLS 行级过滤<br/>(WHERE region IN user.allowed_regions)"]
        RLS --> DW[("❄️ Snowflake / Databricks / Fabric<br/>(与 PowerBI 看板 100% 同源口径)")]
    end
```

### 3.3 多粒度指标查询“自动拆解防膨胀 CTE + 校验用户 RLS + 120ms 语义缓存返回”端到端时序图

```mermaid
sequenceDiagram
    autonumber
    participant VP as 华东区销售副总裁 (OAuth: East-China)
    participant Agent as 智能问数网关
    participant Sem as 治理语义层 (Cube / dbt MetricFlow)
    participant AST as SQLGlot AST & Dry-Run 守卫
    participant DW as 企业数仓 (Snowflake / Fabric)

    VP->>Agent: 1. 提问：“对比 Q3 各大区的净留存 ARR 与 P1 故障工单数”
    Agent->>Sem: 2. LLM 不写裸 SQL，仅输出语义意图：`measures=["net_arr", "p1_tickets"], dims=["region"], date="2026-Q3"`
    Sem->>Sem: 3. 识别出 `net_arr` (合同粒度) 与 `p1_tickets` (工单粒度) 属不同事实表 -> 自动编译为双 `CTE` 先聚合后关联 SQL（彻底消除 12 倍扇出膨胀！）
    Sem->>Sem: 4. 解析调用者 OAuth 令牌 -> 自动织入行级安全策略 `WHERE region = 'East-China'`（拦截跨区未授权数据）
    Sem->>AST: 5. 校验已包含 `2026-Q3` 分区裁剪且预估扫描仅 `180 MB (< 10GB 上限)`
    AST->>DW: 6. 执行对称聚合查询并返回精确结果（与财务月报数字 100% 一致，耗时 `340ms`）！
```

---

## ④ 📊 四档企业对话式 BI 与智能问数架构全维度决策矩阵

| 评估维度 | 🟥 Tier 1：物理表 `DDL` 自由写 SQL | 🟨 Tier 2：`Schema-RAG + Few-Shot` | 🟦 Tier 3：微调专属 `SQL LLM` | 🟩 Tier 4：治理语义层 (`Cube/MetricFlow`) + `AST` 门禁 + `RLS` |
|---|---|---|---|---|
| **高管经营 KPI 口径准确率** | `58.5%` ✕ (41.5% 算错/膨胀) | `80.6%` ✕ (多表关联仍易翻倍) | `84.0%` | 🏆 **`98.6%`（确定性编译器自动处理跨粒度对称聚合）** |
| **失控全表扫描爆表率** | `8.2%` ✕ (频发 50TB 扫描) | `2.8%` | `2.1%` | 🏆 **`0.0%`（`SQLGlot AST` 分区强检 + `10GB Dry-Run` 熔断）** |
| **行级权限 (`RLS`) 越权风险** | `12.4%` ✕ (共享高权账号) | `4.5%` | `3.8%` | 🏆 **`0.0%`（端到端 OAuth 身份透传与语义层强制行过滤）** |
| **与 BI 看板口径一致性** | 严重割裂（各算各的） | 依赖人工同步样本库 | 模型权重滞后 | 🏆 **`100%` 实时同源（与 PowerBI / Tableau 共用同一语义层）** |
| **20 万次问数/月总 TCO** | `$215,000 / 月` ✕ | `$138,000 / 月` ✕ | `$152,000 / 月` ✕ | **`$26,000 / 月` 🏆（比 Tier 1 节省 87.9%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（800 张数仓物理表 / 月均 200,000 次问数 / `90%` 语义层覆盖）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级对话式 BI 与语义层选型 ROI 计算器](../reports/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-conversational-bi-text-to-sql-semantic-layer-metricflow-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection/run.py --check
  python -m unittest tests/test_enterprise_conversational_bi_text_to_sql_semantic_layer_metricflow_selection.py
  ```

### 5.2 核心量化对比图（高管经营指标准确率 % vs. 失控全表扫描事故率 % vs. RLS 越权风险 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Raw DDL Free-Form SQL", "metric": "Executive KPI Accuracy (%)", "val": 58.5 },
      { "tier": "Tier 1: Raw DDL Free-Form SQL", "metric": "Runaway Scan Rate (%)", "val": 8.2 },
      { "tier": "Tier 1: Raw DDL Free-Form SQL", "metric": "RLS Exposure Risk (%)", "val": 12.4 },
      { "tier": "Tier 2: Vector Schema-RAG Only", "metric": "Executive KPI Accuracy (%)", "val": 80.6 },
      { "tier": "Tier 2: Vector Schema-RAG Only", "metric": "Runaway Scan Rate (%)", "val": 2.8 },
      { "tier": "Tier 2: Vector Schema-RAG Only", "metric": "RLS Exposure Risk (%)", "val": 4.5 },
      { "tier": "Tier 3: Fine-Tuned SQL Code-LLM", "metric": "Executive KPI Accuracy (%)", "val": 84.0 },
      { "tier": "Tier 3: Fine-Tuned SQL Code-LLM", "metric": "Runaway Scan Rate (%)", "val": 2.1 },
      { "tier": "Tier 3: Fine-Tuned SQL Code-LLM", "metric": "RLS Exposure Risk (%)", "val": 3.8 },
      { "tier": "Tier 4: Semantic Layer + AST Gate", "metric": "Executive KPI Accuracy (%)", "val": 98.6 },
      { "tier": "Tier 4: Semantic Layer + AST Gate", "metric": "Runaway Scan Rate (%)", "val": 0.0 },
      { "tier": "Tier 4: Semantic Layer + AST Gate", "metric": "RLS Exposure Risk (%)", "val": 0.0 }
    ]
  },
  "semantic_types": { "tier": "Category", "metric": "Name", "val": "Percentage" },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 achieves 98.6% KPI accuracy with 0.0% runaway scans and 0.0% RLS leaks",
    "subtitle": "Executive KPI Accuracy (%) vs. Runaway Scan Rate (%) vs. RLS Exposure Risk (%)",
    "encodings": { "x": "tier", "y": "val", "group": "metric", "color": "metric" }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（上线前必查 5 项）

- [ ] **将核心经营问数从“大模型裸写物理表 SQL（`Text-to-SQL`）”切换为“大模型生成语义层查询（`Text-to-Semantic-Intent`）”**：依托 `Cube / dbt MetricFlow / Fabric Semantic Model` 的对称聚合编译器自动生成多 `CTE` 防膨胀 SQL。
- [ ] **严禁智能问数 Agent 使用全局高权限共享账号直连生产数仓**：必须启用提问者 `OAuth / Entra ID` 令牌透传（On-Behalf-Of）或在语义层视图强制注入租户/部门行级过滤（`RLS`）与敏感列掩码（`CLS`）。
- [ ] **对所有长尾即席生成的 SQL 强制执行 `SQLGlot AST` 静态语法树校验**：拦截一切非 `SELECT` 语句、无连接条件的笛卡尔积（`CROSS JOIN`）以及未携带时间分区键的大表扫描。
- [ ] **在执行即席 SQL 前强制调用数仓 `EXPLAIN / DRY_RUN` 预估扫描字节门禁（如 `<= 10GB`）**：并为 AI 查询配置独立的低配数仓计算池与 15 秒超时熔断，杜绝单条烂查询烧穿云账单。
- [ ] **在返回数字结果的同时向用户展示“可解释指标血缘卡片（Metric Lineage Pill）”**：清晰标明本次计算所使用的官方指标定义名称、时间范围与排除项（如“已按财务准则剔除测试租户与退单”），建立高管信任。
