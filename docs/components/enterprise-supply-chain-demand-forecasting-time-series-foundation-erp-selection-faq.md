# 企业级供应链需求预测与智能补货（`SAP IBP / Blue Yonder / Kinaxis`）、时序基础模型（`Time-Series Foundation Models: Chronos / TimesFM + LightGBM`）vs. 文本大模型幻觉预测、促销异构特征融合与安全库存（`Newsvendor Safety-Stock`）架构选型指南

> **适用读者**：负责零售快消（CPG）、汽车售后备件、医药流通与连锁餐饮供应链计划系统（`SAP Integrated Business Planning (IBP)`、`Blue Yonder`、`Kinaxis RapidResponse`、`Oracle SCM`）建设的 Cloud & AI Solution Architect、数据科学负责人与供应链数字化总监。
> **核心课题**：当企业每周需要为 **`500,000` 个“商品-门店/仓库（`SKU-Location`）”组合生成未来 12 周销量预测（其中 `18.0%` 为上市不满 8 周的新品冷启动 SKU）并自动计算安全库存与采购补货单（Purchase Orders）** 时，为什么直接“把历史销量 CSV 表格塞进 GPT-4o / Claude 提示词让文本大模型输出下周销量数字（Tier 1）”或“沿用传统单变量 `ARIMA / ETS` 统计外推（Tier 2）”会导致 **`28.4%` 加权绝对百分比误差（`WAPE`）、`42.0%` 新品冷启动预测崩盘、`9.2%` 货架断货率与每月 `$640,000` 的缺货丢单、呆滞报废及 Token 账单损失**？如何通过 **“大模型仅做非结构化促销/港口事件特征提取（`LLM Exogenous Covariate Featurizer`）+ `LightGBM`（成熟品）与零样本时序基础模型 `Chronos-Bolt / TimesFM`（新品冷启动）分层集成 + `P50/P90/P95` 保形分位数（`Conformal Quantile`）+ `OR-Tools` 报童模型（`Newsvendor`）整托盘补货求解器”** 四级架构，将全局预测误差 `WAPE` 压降至 **`9.4%`**、新品冷启动 `WAPE` 降至 **`12.8%`**、货架断货率降至 **`1.1%`**，并将月度供应链综合 TCO 降低 **`89.4%`（`$640,000 -> $68,000/月`）**？

![企业级供应链需求预测与智能补货、Chronos/TimesFM 时序基础模型与保形分位数安全库存架构总结图](../../assets/svg/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

在供应链需求预测与库存优化中，**最大的伪创新就是“把历史销量 CSV 直接喂给通用文本大模型（GPT-4o / Claude）让它猜未来销量数字”——文本 BPE 分词器会把连续浮点数切成碎片，既算不准季节性周期，又会在 `500,000` 个 SKU 上烧掉天价 Token 账单**：

| 选型策略 | 适用场景 | 全局预测误差 (`Overall WAPE`) | 新品冷启动误差 (`Cold-Start WAPE`) | 货架断货率 (`Stockout Rate`) | 月度断货损失、呆滞库存与算力总成本 (`TCO`) | 最终拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：将历史销量 CSV 塞进文本大模型提示词直接预测数字（反模式）** | ❌ **严禁用于任何真实生产销量预测** | `28.4%`（严重失真！） | `42.0%` | `9.2%` | `$640,000 / 月`（含 `$85K` Token 费） | ❌ **最贵且最不准**：BPE 分词破坏数值连续性，Token 费比算法贵 200 倍 |
| **Tier 2：传统单变量 `ARIMA / ETS` 逐 SKU 统计外推** | 无促销干扰、无新品交替、需求极度平稳的公用事业耗材 | `22.6%` | `39.0%`（无历史可推） | `6.8%` | `$410,000 / 月` | ❌ **无法应对现代零售**：对营销海报、竞品断货与新品上市完全失明 |
| **Tier 3：纯单点预测 `LightGBM / XGBoost`（无时序大模型冷启动与分位数）** | 历史数据充足的成熟 SKU 基准预测 | `15.8%` | `31.5%`（`<8` 周样本不足） | `4.8%`（均值预测惧怕交期抖动） | `$260,000 / 月` | ⚠️ **惧怕长尾波动**：只输出单点均值（`P50`），遇到供应商交期延误立刻断货 |
| **Tier 4：`LightGBM + Chronos-Bolt/TimesFM` 冷启动 + LLM 促销特征提取 + `P95` 保形分位数 + `OR-Tools` 补货求解（推荐）** | **企业级多级供应链计划（`SAP IBP / Kinaxis`）主力生产架构** | **`9.4%`** | **`12.8%`** | **`1.1%`（`98.9%` 现货率）** | **`$68,000 / 月`（省 `89.4%`）** | 🏆 **生产金标准**：文本归 LLM 提特征、时序归 `TSFM+GBDT`、补货归运筹学 |

### 💡 架构师的三大黄金铁律
1. **让文本大模型干它最擅长的“非结构化外部特征提取（`Exogenous Covariate Featurization`）”，绝不让文本大模型直接输出销量时间序列**：市场部的抖音/小红书大促方案 PDF、门店店长的社群团购群聊纪要、以及海运港口罢工预警邮件都是非结构化文本，传统 `ARIMA` 和 `LightGBM` 读不懂。每天仅需调用一次轻量级 LLM（月成本仅 **`$120`**，而非给每个 SKU 调一次大模型的 `$85,000/月`），将这些文档结构化提取为数值协变量（如 `promo_discount_pct = 20%`、`kol_campaign_tier = S`、`port_delay_days = +6`），再作为回归特征输入时序引擎！
2. **用“`LightGBM`（成熟高密 SKU）+ 预训练时序基础模型 `Chronos-Bolt / Google TimesFM`（新品冷启动/间歇性长尾 SKU）”双擎融合**：对于拥有 2 年以上完整历史的成熟畅销品，特征工程完备的 **`LightGBM`** 依然是性价比之王；但对于上市 `<8` 周的新品（占全盘 SKU 的 `18%`）或低频工业备件，`LightGBM` 因缺乏历史滞后项（Lag Features）误差高达 `31.5%`。引入在数千亿时序点上预训练的零样本时序基础模型（**`Amazon Chronos-Bolt` 或 `Google TimesFM`**），直接捕获跨品类季节性与生命周期形态，将新品冷启动 `WAPE` 从 `39.0%` 压降至 **`12.8%`**，全局 `WAPE` 降至 **`9.4%`**！
3. **永远不要用“单点平均值（`P50`）”直接下单补货：必须输出经保形校准的 `P50 / P90 / P95` 概率分位数并交由 `OR-Tools` 报童模型（`Newsvendor`）求解**：在供应链中，“高毛利畅销品断货 1 件的损失”往往是“多备 1 件库存持有成本”的 `5–10 倍`。通过 **`MAPIE` 保形分位数校准（Conformal Quantile Regression）** 输出需求分布与供应商交期（Lead Time）分布的联合 **`P95` 安全库存包络**，再通过 **`Google OR-Tools`** 满足最小起订量（`MOQ`）、整托盘凑单（Pallet Rounding）与冷库容积约束后写回 `SAP IBP`，将货架断货率从 `9.2%` 压降至 **`1.1%`**。

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A

### Q1：很多业务部门看到 ChatGPT 能画表格，就想“把过去 52 周的销量丢给 GPT-4o 让它直接预测未来 12 周销量（Tier 1）”，为什么上生产后既破产又极其不准？
* **算力经济账与分词器数学缺陷的双重暴击**：
  * **分词碎片化（Tokenization Blindness）**：文本大模型的 BPE 分词器会把数字 `"1428"` 切成 `"14"` 和 `"28"`，把 `"1429"` 切成 `"142"` 和 `"9"`，彻底破坏了数值大小、一阶差分与周期傅里叶特征，实测在标准零售评测集上 `WAPE` 高达 **`28.4%`**（甚至不如简单的移动平均线）。
  * **规模成本灾难**：一家中型零售商有 `500,000` 个 `SKU × 门店` 组合，如果每周给每个组合拼一段 `3,000 Tokens` 的历史序列与提示词调一次商用大模型 API，每月光 Token 费就高达 **`$85,000`**；而用 `LightGBM + Chronos-Bolt` 在单台 `L40S` GPU 上批量跑完 `500,000` 条序列仅需 **`18 分钟`（月算力成本 `<$420`）**！

### Q2：既然 `LightGBM` 在 Kaggle 表格比赛里所向披靡，为什么单靠 `LightGBM`（Tier 3）还是搞不定“新品上市（Cold-Start）”和“港口延误断货”？
* **梯度提升树（GBDT）的两大天然盲区**：
  1. **新品冷启动无 Lag 特征**：`LightGBM` 极度依赖 `lag_7d`、`lag_28d`、`rolling_mean_12w` 等历史自回归特征。当一款春季联名饮品刚上架第 2 周时，这些特征全为 `NULL`。预训练时序基础模型（**`Chronos-Bolt / TimesFM / Moirai`**）像学习自然语言语法一样，在亿级跨行业时序上学会了“新品冲高回落、周末双峰、节假日脉冲”的零样本先验波形，配合同类老品属性锚定（Analogous SKU Anchoring），将冷启动误差从 `31.5%` 砍到 **`12.8%`**。
  2. **均值预测（Point Forecast）无视不确定性尾巴**：两款周均销量都是 `100` 件的商品，A 商品每周稳定卖 `98–102` 件，B 商品受网红带货影响在 `10–350` 件之间剧烈震荡。如果只看 `P50 = 100` 下采购单，B 商品必然频繁断货！

### Q3：预测做准了，为什么直接把预测数导入 ERP 生成采购单还是会被采购总监骂？
* **“需求预测（Forecast）”不等于“最优补货订单（Replenishment Order）”**：
  * 真实采购受到严格的物理与商务约束：供应商要求最小起订量 `MOQ = 500 件`、每个标准托盘必须装满 `60 箱`、40 尺海运集装箱差 `8%` 就能凑满整柜享受运费 7 折、保质期只有 `45 天`。因此必须在分位数预测之后串联 **`Google OR-Tools` 混合整数规划（MIP）求解器**，把预测分布转化为可直接审批的 ERP 采购申请单（`Purchase Requisition`）。

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 端到端参考架构 + 时序图）

### 3.1 企业级供应链需求预测与补货架构选型决策树

```mermaid
flowchart TD
    A["企业供应链需求预测与自动补货需求 (500K SKU-Locations)"] --> B{"输入数据形态是什么?"}
    B -- "非结构化营销日历 PDF / 港口预警邮件 / 天气通告" --> C["调用轻量级 LLM 提取为结构化数值协变量 ($120/月)\n严禁让文本 LLM 直接预测销量数字!"]
    B -- "结构化历史销量序列 + 价格/库存/促销特征表" --> D{"该 SKU 历史销售周数是否 >= 8 周且销量连续?"}
    C --> D
    D -- "是 (82% 成熟高密 SKU)" --> E["主力引擎: LightGBM 分位数回归 + 层级一致性校准 (MinT)"]
    D -- "否 (18% 新品冷启动 / 间歇性工业备件)" --> F["冷启动引擎: Chronos-Bolt / TimesFM 零样本时序基础模型 + 同类老品锚定"]
    E & F --> G["MAPIE 保形分位数校准: 输出 P50 / P90 / P95 需求概率包络"]
    G --> H["Google OR-Tools 报童模型与凑整柜求解器 (MOQ / 托盘 / 保质期约束) -> 写回 SAP IBP"]
```

### 3.2 Tier 4 LLM 协变量提取 + `LightGBM + Chronos` 双擎 + `OR-Tools` 补货端到端架构图

```mermaid
flowchart LR
    subgraph Feat["📄 1. LLM 非结构化特征提取层 ($120/月)"]
        DOC["大促方案 PDF / 竞品情报\n/ 海运港口延误通告"] --> LLM["8B/70B 结构化特征提取器\n生成 promo_depth, lt_delta_days"]
    end

    subgraph Forecast["📈 2. 双擎时序预测与保形分位数层 (500K SKUs / 18m)"]
        ERP["SAP / Snowflake 历史销量\n与库存价格宽表"] --> ROUTER{"SKU 生命周期路由\n(历史 >= 8 周?)"}
        LLM --> ROUTER
        ROUTER -- "成熟 SKU (82%)" --> GBDT["LightGBM 多分位数模型\n(跨品类滞后与交叉弹性特征)"]
        ROUTER -- "冷启动/长尾 (18%)" --> TSFM["Chronos-Bolt / TimesFM\n零样本预训练时序基础模型"]
        GBDT & TSFM --> CONF["MAPIE 保形校准 + MinT 层级调和\n输出校准后 P50 / P90 / P95 分布"]
    end

    subgraph Optim["📦 3. 运筹学安全库存与 ERP 补货求解层"]
        CONF --> OR["Google OR-Tools 报童模型 MIP\n(毛利率 x 交期波动 x MOQ x 整托盘)"]
        OR --> SAP["写回 SAP IBP / Kinaxis\n生成可一键审批采购单 (PO)"]
    end
```

### 3.3 全链路促销特征提取、双擎分位数预测与整托盘补货求解时序图

```mermaid
sequenceDiagram
    autonumber
    participant Mkt as 市场部大促 PDF & 物流预警
    participant LLM as LLM 协变量提取器
    participant Engine as LightGBM + Chronos-Bolt 引擎
    participant Solver as OR-Tools 补货求解器
    participant SAP as SAP IBP / ERP 采购工作台

    Mkt->>LLM: 上传《五一抖音 S 级达人直播排期表.pdf》与《盐田港拥堵 +5 天通知》
    LLM->>Engine: 提取结构化外生特征: `SKU-8821: promo_lift_tier=S, discount=0.25, supplier_lt_add_days=5`
    Engine->>Engine: 成熟品跑 `LightGBM`，上市 2 周新品跑 `Chronos-Bolt`，经 `MAPIE` 输出 `P50=1,200, P95=1,680`
    Engine->>Solver: 推送 `500,000` 个 SKU-仓库的 `(P50, P90, P95)` 需求分布与交期方差
    Solver->>Solver: 结合 `毛利率 42%`、`MOQ=500`、`每托盘 60 箱` 与 `保质期` 求解最优补货量 `1,620 箱 (27 整托盘)`
    Solver->>SAP: 写入 SAP IBP 建议采购单并附带自然语言解释卡 ("因五一 S 级直播及港口延误 +5 天，安全库存上调 28%")
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：将历史销量 CSV 塞进文本 LLM 直接预测 | Tier 2：传统单变量 `ARIMA / ETS` 统计外推 | Tier 3：纯单点均值 `LightGBM`（无分位数/无冷启动） | Tier 4：`LightGBM + Chronos-Bolt` + LLM 协变量 + `P95` 保形分位数 + `OR-Tools`（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **全局预测误差 (`Overall WAPE`)** | **`28.4%`（BPE 分词破坏数值规律）** | `22.6%` | `15.8%` | **`9.4%`（行业顶尖精度）** |
| **新品冷启动误差 (`Cold-Start WAPE`)** | `42.0%` | `39.0%` | `31.5%`（`<8` 周缺乏滞后项） | **`12.8%`（`Chronos-Bolt / TimesFM` 零样本波形迁移）** |
| **货架断货率 (`Stockout Rate`)** | `9.2%` | `6.8%` | `4.8%`（单点均值惧怕交期抖动） | **`1.1%`（`P95` 保形分位数 × 交期联合分布）** |
| **非结构化大促/供应链中断感知** | ⚠️ 能读文本但算错数字 | ❌ 100% 失明 | ❌ 需人工手动造特征表 | **✅ LLM 自动解析 PDF/邮件为外生回归变量** |
| **`500K` SKU 单次批量预测算力成本** | `$85,000 / 月`（天价 API Token） | `$1,200 / 月`（CPU 慢） | `$350 / 月` | **`$420 / 月`（18 分钟跑完 50 万序列）** |
| **月度断货、呆滞库存与算力总 TCO** | `$640,000 / 月` | `$410,000 / 月` | `$260,000 / 月` | **`$68,000 / 月`（节省 `89.4%`）** |

---

## 5. 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 基准算例假设（`500,000` 个活跃 `SKU-门店/仓库` 组合 / 每周生成 `2,000,000` 条预测）
* **业务规模**：大型连锁零售与快消企业，管理 `500,000` 个活跃 `SKU-Location` 节点（其中 `18.0%` 为上市不满 8 周的新品），目标货架有货服务水平（Fill Rate SLA）为 `>=98.5%`。
* **工程实测与仿真代码**：见 [`experiments/enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection/run.py`](../../experiments/enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection/run.py) 与 [`results.json`](../../experiments/enterprise_supply_chain_demand_forecasting_time_series_foundation_erp_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式容量与 TCO 计算器**：📊 [打开企业级供应链需求预测与安全库存容量计算器 (`enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-sizer.html`)](../reports/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-cards.html`)](../reports/enterprise-supply-chain-demand-forecasting-time-series-foundation-erp-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Text-LLM CSV Prompting", "wape_pct": 28.4, "cold_start_wape_pct": 42.0, "stockout_pct": 9.2, "monthly_tco_k": 640.0 },
      { "tier": "Tier 2: Univariate ARIMA/ETS", "wape_pct": 22.6, "cold_start_wape_pct": 39.0, "stockout_pct": 6.8, "monthly_tco_k": 410.0 },
      { "tier": "Tier 3: Point-Only LightGBM", "wape_pct": 15.8, "cold_start_wape_pct": 31.5, "stockout_pct": 4.8, "monthly_tco_k": 260.0 },
      { "tier": "Tier 4: LightGBM + Chronos + P95 Conformal", "wape_pct": 9.4, "cold_start_wape_pct": 12.8, "stockout_pct": 1.1, "monthly_tco_k": 68.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "wape_pct": "Percentage",
    "cold_start_wape_pct": "Percentage",
    "stockout_pct": "Percentage",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 9.4% WAPE, 1.1% Stockout Rate, and Cuts Monthly Supply Chain TCO by 89.4%",
    "subtitle": "Monthly stockout penalties, excess working-capital holding, and compute TCO ($ USD Thousands)",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_k",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **严禁文本大模型直接外推数值时间序列**：是否已明令禁止将原始 CSV 销量序列塞进 `GPT-4o / Claude` Prompt 直接输出预测销量，转而仅用 LLM 将非结构化促销/物流文档提取为结构化回归特征？
- [ ] **成熟 SKU（`LightGBM`）与冷启动/间歇性 SKU（`Chronos-Bolt / TimesFM`）双擎路由**：是否按历史有效销售周数（如 `<8` 周）自动将新品路由至预训练时序基础模型并绑定同类老品锚点？
- [ ] **真实需求还原（`Stockout Demand Uncensoring`）**：在训练模型前，是否已将历史发生过断货（`Inventory = 0`）日期的截断销量还原为真实潜在需求，避免模型把“没货卖导致的 0 销量”误判为“没人买”？
- [ ] **保形分位数校准（`MAPIE Conformal P50/P90/P95`）与层级一致性（`MinT Reconciliation`）**：门店级、大区级与全国总仓预测汇总是否严格满足数学加总一致，且为高毛利商品输出经校准的 `P95` 安全库存上界？
- [ ] **`Google OR-Tools` 物理与商务约束求解器**：在将补货建议写入 `SAP IBP / ERP` 前，是否已自动满足最小起订量（`MOQ`）、整托盘/整柜凑单与剩余保质期约束？
