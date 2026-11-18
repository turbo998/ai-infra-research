# 企业级连锁酒店与文旅度假区（Hospitality & Resort Chain）动态收益管理（`RMS`）、超售拒载（`Walk-Risk`）非对称优化、会展团房洗房预测与客房保洁（`Housekeeping MILP`）联动调度 Agent 架构选型指南：传统淡旺季固定房价+按楼层顺扫 vs. 纯 LLM 抓取竞品跟跌降价 vs. 未解截断的历史入住率回归 vs. `EM` 需求解截断连住影子价格 `DP` + 二项式超售临界分位求解 + 航班/网约车 `ETA` 联动客房调度 Agent

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-hospitality-hotel-revenue-management-overbooking-housekeeping-agent-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、跨国酒店集团/主题乐园度假区首席商务官（CBO / VP of Revenue Management）、物业管理系统（`Opera Cloud / Mews PMS`）工程负责人与酒店运营总监的生产落地指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_hospitality_hotel_revenue_management_overbooking_housekeeping_agent_selection/results.json`](../../experiments/enterprise_hospitality_hotel_revenue_management_overbooking_housekeeping_agent_selection/results.json)。

![企业级连锁酒店动态收益管理与超售控制及客房调度 Agent 架构选型总结图](../../assets/svg/enterprise-hospitality-hotel-revenue-management-overbooking-housekeeping-agent-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在运营 **120 家中高端与奢华连锁酒店（共 `36,000` 间客房 Keys、日均 `24,000` 次入住/退房翻台、基准平均房价 `ADR $210`）** 时，酒店集团收益管理总监与运营副总裁面临三大“利润蒸发黑洞”：
1. **用爬虫 + 通用大模型（Tier 2）无脑跟跌周边竞品 OTA 房价，引发 `-3.2% RevPAR` 恶性价格战，且“周二单晚高价客”挤走了“周一至周四连住 4 晚大客户”**：当周二有大型科技展导致只剩 40 间房时，Tier 1/2 欣然接下 `$260` 的周二单晚散客直至周二售罄（Sold Out），结果导致愿意花 `$230/晚 × 4 晚 = $920`（周一至周四连住）的商务客因周二没房而被迫去住竞品——造成周一、周三、周四房间空置，每间房净亏 **`$660`**！
2. **用被 `100%` 满房上限截断（Censored）的历史入住率训练回归模型（Tier 3），且固定 `10%` 超售经验法则造成恶劣的“半夜赶客（Walking a Guest）”或空房浪费**：OTA 免费取消订单比例高达 32%，若按固定 10%（+50 间）超售，一旦遇到暴雨航班取消率骤降，当晚就会多出 14 名无房可住的钻石卡会员（每次拒载赔偿同城五星房费+交通+积分高达 **`$650`** 外加差评风暴）；
3. **客房阿姨按房号（201, 202, 203...）机械顺扫，导致 `29.5%` 提前到店客人苦坐大堂 90 分钟**：未将手机 App 移动端值机（Mobile Check-In）的机场落地 `ETA` 与房间属性（高楼层/安静/大床）接入派工优化。

在 **120 家酒店、36,000 间客房、日均 24,000 间次翻台** 的企业级基准规模下，四档酒店收益管理与运营调度架构的量化对决如下：

| 方案档位 | 核心架构组合 | 单房收益提升 (`RevPAR Uplift`) | 超售拒载或空房浪费率 | 提前到店大堂苦等超时率 (`>30m`) | 会展团房虚占 (`Wash`) 预测误差 | 月度空房浪费、拒载赔偿与加班总损耗 (`TCO`) | 生产推荐度 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：传统淡旺季 BAR 固定阶梯 + 10% 固定超售 + 按层顺扫** | 静态淡旺季价格表 + 一刀切超售 10% + 保洁员按房号从头扫到尾 | `0.0% (基线)` | `8.4%` ✕ | `29.5%` ✕ *(大堂爆满)* | `22.0%` ✕ | `$540,000 / 月` ✕ | ⛔ 单晚客挤死连住客，大堂天天吵架 |
| **🟨 Tier 2：纯大模型爬虫跟价 (`Competitor-Undercut LLM`)** | 抓取周边 OTA 价格让 LLM 永远比竞品便宜 $5（无连住影子价格与解截断） | `-3.2%` ✕ *(恶性降价)* | `11.2%` ✕ | `27.0%` ✕ | `25.5%` ✕ | `$635,000 / 月` ✕ | ⛔ 自己 92% 满房还跟着空虚竞品打折 |
| **🟦 Tier 3：常规历史入住率时序回归 (未解截断/无连住 DP)** | 直接拿封顶 100% 的历史入住率做回归 + 对称均方误差超售 | `+4.8%` | `4.1%` ✕ | `16.5%` ✕ | `11.8%` ✕ | `$265,000 / 月` ✕ | ⚠️ 把“卖光了”当成“只有100%需求” |
| **🟩 Tier 4：`EM` 需求解截断连住影子价格 `DP` + 非对称超售求解器 + 到店 `ETA` 客房 `MILP` Agent** | **`EM` 算法需求解截断 + 连住天数 (`LOS`) 网络动态规划影子价格 (`Bid Price`) + 报童模型非对称拒载分位数 + 移动端 `ETA` 联动保洁 `MILP`** | **`+14.6%` 🏆** *(+17.8 pts)* | **`0.3%` 🏆** *(零VIP拒载)* | **`0.8%` 🏆** *(-28.7 pts)* | **`2.1%` 🏆** *(提前21天释房)* | **`$48,000 / 月` 🏆** *(-91.1%)* | 🌟 **现代酒店集团与度假区唯一生产级架构** |

> 💡 **架构师一句话选型铁律**：**“酒店收益管理决不能做‘比隔壁酒店便宜 5 块钱’的爬虫跟屁虫，也决不能直接拿封顶 100% 的历史入住率去训练模型——必须先用 `EM（期望最大化）` 算法还原满房被拒之门外的真实无约束需求，再用网络动态规划算出每一天的房间影子门槛价（`Shadow Bid Price`）保护跨天连住客，用报童临界分位公理平衡 `$210 空房损失` 与 `$650 拒载赔偿`，并把客人的下飞机 `ETA` 实时注入客房阿姨的 `MILP` 派单队列！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大酒店收益与运营痛点）

### Q1：为什么周二只剩 40 间房时，拒绝 `$260` 的周二单晚客人、把房间留给 `$230/晚` 的连住 4 晚客人反而多赚 `$660`？什么是连住网络影子价格（`Network LOS Bid Price`）？
- **大白话与第一性原理**：
  1. **单晚峰值瓶颈挤出效应（Displacement Effect）**：城市周二有大型展会（满房日），但周一、周三、周四是平淡日（入住率仅 65%）。如果收益系统只看单日房价，觉得 `$260 > $230` 就把周二最后 40 间房全卖给只住周二 1 晚的客人，那么想订“周一至周四连住 4 晚（总房价 `$230 × 4 = $920`）”的商务客就会因为周二断房而整单流失！结果酒店只赚了周二的 `$260`，周一、周三、周四连续空置 3 晚，**每间房白白少赚 `$920 - $260 = $660`**！
  2. **Tier 4 的 `EM 需求解截断 + 网络连住影子价格（Shadow Bid-Price Hurdle）`**：
     - 首先用 **期望最大化（`Expectation-Maximization, EM`）解截断算法**，根据周二提前 5 天售罄的预订斜率，还原出周二如果不限量的真实潜在需求其实是 `148%`（而非历史报表上干瘪的 `100%`）。
     - 随后通过网络动态规划（`Network DP`）算出周一至周四每一晚的边际机会成本（即**影子价格 Shadow Bid Price**）：$\pi_{\text{Mon}} = \$110, \pi_{\text{Tue}} = \$285, \pi_{\text{Wed}} = \$195, \pi_{\text{Thu}} = \$140$。
     - **确定性接单门禁规则**：任何入住请求 $(t, \text{LOS}=L)$，只有当**总房费 $\ge \sum_{d=t}^{t+L-1} \pi_d$** 时才放行！因此：
       - 周二单晚 `$260 < \pi_{\text{Tue}} (\$285)` $\to$ **自动关房拒接单晚散客**；
       - 周一至周四 4 晚连住总价 `$920 \ge \$110 + \$285 + \$195 + \$140 = \$730` $\to$ **自动放行接单**！直接带动全集团 **`RevPAR +14.6%`** 净增长！

### Q2：面对 OTA 渠道高达 32% 的免费取消率与会议团房 29% 的“洗房缩水（Group Wash）”，如何精准设定超售垫子（Overbooking Pad）做到既不空房又不赶客？
- **非对称报童临界分位（Asymmetric Newsvendor Critical Fractile）+ 团房存活曲线**：
  1. **空一间房亏 `$210` vs. 赶走一位 VIP 亏 `$650`**：普通的回归模型（Tier 3）用对称的均方误差（MSE）预测取消人数，把“多留一间空房”与“把超售客人半夜赶去别的酒店（Walk）”视为同等代价。
  2. Tier 4 对在手预订中每一笔订单按 `(渠道 × 提前预订天数 × 是否含早/不可退 × 航班动态)` 拟合非齐次二项生存分布 $F_{\text{no-show/cancel}}(Q)$，并严格按**非对称报童临界分位公式**求解最优超售间数 $Q^*$：
     $$F_{\text{cancel}}(Q^*) = \frac{C_{\text{spoilage}}}{C_{\text{spoilage}} + C_{\text{walk}}} = \frac{210}{210 + 650} = 0.244$$
     算出当晚精确超售 **`+19 间`**（而非盲目的 10% = +50 间），将空房与拒载总损失率从 `8.4%` 压降至 **`0.3%`**！
  3. **会展团房洗房（Group Block Wash）提前 21 天释放**：针对医学协会提前半年锁定的 200 间会议团房，通过追踪参会人员名单录入进度（Rooming-List Pacing）对比历史同类协会存活曲线，提前 **21 天** 精准预判将有 `29%（58 间）` 缩水不来，立即将其中 **54 间** 释放回高价散客池售卖，杜绝拖到入住前 7 天截止日（Cutoff Date）才挥泪甩卖！

### Q3：为什么下午 13:30 大堂里会有 85 位客人排队骂街等房？保洁派单 `MILP` 如何将等待超时率从 `29.5%` 砍到 `0.8%`？
- **告别“按楼层顺扫”，实现“移动端落地 `ETA` + 房间属性（ABS）动态对齐”**：
  - 传统客房部领班早上给每位保洁阿姨发一张固定楼层表（从 201 扫到 215，Tier 1），结果阿姨正在慢条斯理打扫今晚 22:00 才到店的 202 房，而已经在虹桥机场坐上网约车、18 分钟后（13:30）就到大堂且购买了**“高楼层+远离电梯+行政大床”属性包（Attribute-Based Selling）**的 85 位客人，对应的 1804 房却还没开始扫！
  - Tier 4 每 10 分钟重求解一次**多班次保洁路径混合整数规划（`MILP`）**：将客人手机 App 的实时预计到店时间（`Mobile ETA`）、延住/延迟退房状态与跨楼层电梯耗时耦合，动态向阿姨手环推送下一间最紧急的目标房号（`"RM_1804_HIGH_KING"`），将提前到店大堂苦等超时率从 `29.5%` 压降至 **`0.8%`**！

---

## ③ 🧭 直观可视化图解（决策树 + 收益与保洁联动架构图 + 展会高峰调度时序图）

### 3.1 企业级连锁酒店动态收益管理（RMS）与运营调度架构决策树

```mermaid
flowchart TD
    A["企业级连锁酒店动态收益管理 (RMS) 与客房运营<br/>(120 家酒店 + 36,000 间客房 + 日均 24,000 次入住翻台)"] --> B{"是否仍用固定淡旺季价格表<br/>或用爬虫 + LLM 无脑比周边竞品便宜 $5?"}
    B -- "是 (固定价格或无脑跟跌)" --> C["⛔ 价格战与连住客流失 (Tier 1 / Tier 2)<br/>RevPAR 倒退 -3.2%，单晚高峰客挤走 4 晚连住大单，且 29.5% 客人大堂苦等"]
    B -- "引入历史入住率预测模型" --> D{"是否执行了 EM 满房需求解截断、连住影子价格 DP、<br/>非对称拒载分位数与移动端 ETA 保洁 MILP?"}
    D -- "否 (直接拿 100% 封顶历史入住率回归)" --> E["⚠️ 满房需求低估与超售赶客 (Tier 3)<br/>漏算团房 29% 缩水洗房，且对称误差导致高峰期 VIP 被迫拒载转店"]
    D -- "EM 解截断 + 连住 DP + 非对称超售 + 保洁 MILP" --> F["🟩 Tier 4: EM 解截断连住影子价格 DP + 二项式超售求解 + 到店 ETA 保洁 MILP 🏆<br/>(+14.6% RevPAR | 0.3% 拒载/空置 | 0.8% 大堂等待超时 | 节省 91.1% 损耗)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `EM` 解截断连住 `DP` + 非对称超售 + 移动端 `ETA` 客房 `MILP` 架构图

```mermaid
flowchart LR
    subgraph Demand["1. EM 解截断与团房洗房预测层"]
        PMS["Opera/Mews PMS 预订流<br/>+ 实时搜索转化率 (Look-to-Book)"] --> EM["📈 EM 无约束需求解截断引擎<br/>还原 100% 满房日真实需求 (148%)"]
        Group["会议团房名单录入进度"] --> Wash["📉 团房洗房 (Wash) 存活曲线<br/>提前 21 天释放 54 间虚占房"]
    end

    subgraph Pricing["2. 连住影子价格 DP 与非对称超售"]
        EM --> DP["⚖️ 网络连住 (LOS) 动态规划<br/>输出每日影子门槛价 π_d<br/>关停单晚低效散客，保护 4 晚连住"]
        Wash --> DP
        Cancel["OTA 渠道 × 提前期取消概率"] --> OB["🛡️ 二项式报童临界分位求解器<br/>F(Q*) = 210/(210+650) = 0.244<br/>精准超售 +19 间 (0.3% 拒载/空置)"]
    end

    subgraph Ops["3. 到店 ETA 联动客房保洁 MILP"]
        ETA["客人移动端值机 + 航班/打车 ETA<br/>+ ABS 选房属性 (高楼层/静音)"] --> MILP["🧹 38 名保洁员动态派单 MILP<br/>优先抢扫 18 分钟后到店房 (0.8% 超时)"]
    end
```

### 3.3 从“周二展会剩余 40 间房防挤出”到“13:30 提前到店高峰零等待入住”时序图

```mermaid
sequenceDiagram
    autonumber
    participant OTA as OTA 预订网关 & 会议团房池
    participant RMS as EM 解截断 + 连住影子价格 DP 引擎
    participant OB as 非对称超售分位求解器
    participant MILP as 移动端 ETA 联动客房保洁 MILP
    participant Guest as 4 晚连住商务客 (13:30 抵达大堂)

    OTA->>RMS: 周二展会仅剩 40 间房：收到订单 A（周二单晚 $260）与订单 B（周一至周四 4 晚 × $230 = $920）
    RMS->>RMS: 计算每日影子门槛价：π_Mon=$110, π_Tue=$285, π_Wed=$195, π_Thu=$140 (4晚合计门槛 $730)
    RMS-->>OTA: 🚫 自动拦截单晚订单 A ($260 < $285)，✅ 自动放行 4 晚连住订单 B ($920 >= $730)，单房多赚 $660！
    OB->>RMS: 提前 21 天识别医学会 200 间团房将缩水 29%，释放 54 间房并设定全店最优超售垫子 +19 间
    Guest->>MILP: 入住当日 13:12，客人通过 App 发送虹桥机场上车通知（预计 18 分钟后抵店，偏好高楼层安静大床）
    MILP->>MILP: 重排 38 名保洁员任务队列：暂停今晚 22:00 到店的 202 房，优先派单打扫并质检 1804 高区安静大床房
    MILP-->>Guest: 🏆 13:30 客人踏入大堂瞬间手机蓝牙钥匙已激活，0 分钟排队直接上楼入住！
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：固定淡旺季 + 10% 超售 + 顺扫 | 🟨 Tier 2：纯 LLM 竞品跟跌降价 | 🟦 Tier 3：未解截断历史入住率回归 | 🟩 Tier 4：`EM` 连住 `DP` + 非对称超售 + 保洁 `MILP` |
|---|---|---|---|---|
| **集团单房收益增幅 (`RevPAR Uplift`)** | `0.0%`（静态基线） | `-3.2%` ✕（满房前盲目跟跌打折） | `+4.8%`（单晚客仍挤走连住客） | 🏆 **`+14.6%`（`EM` 解截断 + 连住影子门槛价 $\pi_d$）** |
| **超售拒载 (`Walk`) 或空房浪费率** | `8.4%` ✕（固定 10% 超售大起大落） | `11.2%` ✕ | `4.1%` ✕（对称 MSE 忽视赶客高昂代价） | 🏆 **`0.3%`（二项生存分布 + `0.244` 报童临界分位）** |
| **会展团房洗房 (`Group Wash`) 预测误差** | `22.0%` ✕（死守到 T-7 截止日才甩卖） | `25.5%` ✕ | `11.8%` | 🏆 **`2.1%`（名单录入斜率存活曲线提前 21 天释房）** |
| **提前到店客人大堂苦等超时率 (`>30m`)** | `29.5%` ✕（按房号机械顺扫） | `27.0%` ✕ | `16.5%` ✕（收益与客房派单脱节） | 🏆 **`0.8%`（实时航班/打车 `ETA` 驱动保洁 `MILP`）** |
| **120 家酒店月度空置、拒载与加班总损耗** | `$540,000 / 月` | `$635,000 / 月` | `$265,000 / 月` | **`$48,000 / 月` 🏆（节省 91.1%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（120 家酒店 / 36,000 间客房 / 日均 24,000 间次翻台）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级连锁酒店动态收益管理与超售及保洁调度定容计算器](../reports/enterprise-hospitality-hotel-revenue-management-overbooking-housekeeping-agent-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-hospitality-hotel-revenue-management-overbooking-housekeeping-agent-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_hospitality_hotel_revenue_management_overbooking_housekeeping_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_hospitality_hotel_revenue_management_overbooking_housekeeping_agent_selection.py
  ```

### 5.2 核心量化对比图（单房收益提升 RevPAR Uplift % vs. 提前到店大堂苦等超时率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Static BAR + FIFO", "metric": "RevPAR Uplift (%)", "pct": 0.0 },
      { "tier": "Tier 1: Static BAR + FIFO", "metric": "Lobby Wait Breach (%)", "pct": 29.5 },
      { "tier": "Tier 2: Competitor-Undercut LLM", "metric": "RevPAR Uplift (%)", "pct": -3.2 },
      { "tier": "Tier 2: Competitor-Undercut LLM", "metric": "Lobby Wait Breach (%)", "pct": 27.0 },
      { "tier": "Tier 3: Censored Occupancy Reg", "metric": "RevPAR Uplift (%)", "pct": 4.8 },
      { "tier": "Tier 3: Censored Occupancy Reg", "metric": "Lobby Wait Breach (%)", "pct": 16.5 },
      { "tier": "Tier 4: EM LOS DP + Walk + MILP", "metric": "RevPAR Uplift (%)", "pct": 14.6 },
      { "tier": "Tier 4: EM LOS DP + Walk + MILP", "metric": "Lobby Wait Breach (%)", "pct": 0.8 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 lifts portfolio RevPAR by +14.6% while cutting early check-in lobby wait breaches to 0.8%",
    "subtitle": "Comparison across 120 hotel properties and 36,000 managed room keys",
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

- [ ] **历史满房日数据必须先经过 `EM（期望最大化）` 或生存分析解截断（Demand Detruncation）**：严禁把受限于物理房间上限的 `100%` 历史入住率直接当作真实需求上限喂给预测模型。
- [ ] **全面启用连住天数（`Length-of-Stay, LOS`）网络动态规划影子门槛价（`Shadow Bid Prices`）**：在展会/节假日高峰期自动关闭低收益单晚散客房仓，保护跨肩期（Shoulder Nights）的高价值多晚连住订单。
- [ ] **超售垫子（Overbooking Pad）必须按非对称报童临界分位 $C_{\text{spoilage}} / (C_{\text{spoilage}} + C_{\text{walk}})$ 动态逐日求解**：对不可退预付单、免费取消 OTA 单与航司延误信息分别赋予独立生存概率，严禁使用固定 `10%` 经验比例。
- [ ] **会展与团队预留房（Group Blocks）部署名单录入斜率（Pickup Pacing）洗房预测器**：在入住前 14~21 天自动识别并提前释放虚占房源回流散客高价池。
- [ ] **打通 PMS 移动端客人预计到店时间（`Mobile ETA`）、选房属性（`ABS`）与客房部保洁派单 `MILP`**：彻底废除按楼层房号死板顺扫，实现“人到房净、零等待拿房”。
