# 企业级汽车售后备件供应链与 4S 门店服务网络（Automotive Aftermarket & Dealership `DMS`）、慢动件间歇性需求预测（`TSB / Croston` 多级调拨）、17 位 `VIN` 码拆车件定损 `VLM` 与维修工位排程 `MILP` Agent 架构选型指南：传统 DMS 移动平均补货+拆解后补单+先到先上举升机 vs. 纯 VLM 看图订件 vs. 普通零售正态分布预测+2D 分割 vs. `TSB` 间歇需求多级共享池 + 17 位 `VIN EPC` 精确锁定 + 3D 暗伤预测 `VLM` 与到货齐套工位 `MILP` Agent

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-automotive-aftermarket-parts-croston-collision-vlm-bay-milp-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、主机厂（OEM）售后备件供应链总监、大型汽车经销商集团（`CDK Global / Tekion / 雷诺兹 DMS`）售后服务副总裁及车险定损中心（`CCC / Audatex`）架构师的生产落地指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection/results.json`](../../experiments/enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection/results.json)。

![企业级汽车售后备件 TSB 预测、17位 VIN 碰撞定损 VLM 与工位排程 MILP Agent 架构总结图](../../assets/svg/enterprise-automotive-aftermarket-parts-croston-collision-vlm-bay-milp-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在管理 **180 家品牌 4S 经销店 + 4 座区域备件分拨中心（`PDC`）、常备 `145,000` 种原厂售后配件 SKU（其中高达 `78%` 属于每年仅消耗 0~3 件的慢动/长尾间歇性配件），并每日接待 `2,800` 台维保与事故钣喷车辆（`RO`）** 的汽车售后集团中，配件、定损与车间调度存在三大出血点：
1. **拿快消零售的移动平均或高斯回归（Tier 1 / Tier 3）去管 `78%` 的汽车慢动件，造成“缺货停驶（`VOR` 高达 `18.4%`）”与“三年卖不掉报废死库存（`21.6%`）”两头挨耳光**：某款冷门转向节一年只坏 1 次，传统 DMS 在卖掉 1 件后立刻按移动平均补货 2 件，结果在门店货架上吃灰 3 年直接变成呆滞报废资产！
2. **钣喷事故车仅凭外观照片订件（Tier 1 / Tier 2），拆开保险杠才发现里面的防撞梁、吸能盒和毫米波雷达支架全碎了（追加拆检补单率 `Supplement Rework` 高达 `26.5%~31.2%`），且纯看图 VLM 分不清同款车的高配矩阵激光大灯与低配卤素大灯，导致 `9.6%` 订错配件号**！
3. **车间先到先上举升机（FIFO，Tier 1）：配件还没到齐就把事故车吊上举升机拆散，一旦缺一颗雷达支架，这台车就像钉子户一样把日均产值 `$1,200` 的黄金工位死死霸占 48 小时**！

在 **184 个售后网点/分拨仓、14.5 万个配件 SKU、日均 2,800 张维修工单** 的集团基准规模下，四档汽车售后备件、智能定损与工位排程架构的量化对决如下：

| 方案档位 | 核心架构组合 | 缺件停驶率 (`VOR Stockout`) | 呆滞死库存报废率 (`Obsolescence`) | 拆检追加补单与返工率 (`Supplement`) | 订错 `VIN` 配件号装车逃逸率 | 月度缺件、死库存与霸占工位总损失 (`TCO`) | 生产推荐度 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：传统 DMS 移动平均 Min/Max + 拆解后补单 + FIFO 排工位** | 6个月移动平均安全库存 + 技师拆车后才补订暗伤件 + 进店即上举升机 | `18.4%` ✕ | `21.6%` ✕ *(偶发销量触发盲补)* | `26.5%` ✕ *(拆开才知里面碎了)* | `4.2%` ✕ *(手工查错改款件)* | `$840,000 / 月` ✕ | ⛔ 满仓呆滞件却缺常用件，举升机被拆散车堵死 |
| **🟨 Tier 2：纯大模型看手机照片定损与自由文本订件 (`Prompt VLM`)** | 把碰撞照片喂给通用 VLM 直接输出零件号与工时预约 | `15.8%` ✕ | `19.4%` ✕ | `31.2%` ✕ *(漏判内部结构件)* | `9.6%` ✕ *(高低配大灯张冠李戴)* | `$950,000 / 月` ✕ | ⛔ 凭外观猜不出内部配置码，错件退货泛滥 |
| **🟦 Tier 3：标准零售连续正态需求 ML + 2D 外观掩码分割** | 零售 ARIMA/XGBoost（假设连续需求）+ 2D 划痕分割（无 17 位 VIN 树） | `7.4%` ✕ | `11.2%` ✕ *(零销量期不衰减)* | `12.8%` ✕ *(缺能量暗伤图谱)* | `1.9%` ✕ | `$380,000 / 月` ✕ | ⚠️ 无法处理 78% 间歇慢动件与覆盖件下方暗伤 |
| **🟩 Tier 4：`TSB` 间歇需求多级池化 + 17位 `VIN EPC` + 3D 暗伤 `VLM` + 齐套工位 `MILP`** | **`TSB` 概率衰减模型 + 区域仓 4h 横向调拨 `MILP` + 3D 凹陷暗伤贝叶斯图谱 + 17位 `VIN PR码 EPC AST` + 齐套扫码上工位 `MILP`** | **`0.6%` 🏆** *(-96.7%)* | **`1.8%` 🏆** *(-19.8 pts)* | **`1.9%` 🏆** *(-24.6 pts)* | **`0.0%` 🏆** *(零订错件)* | **`$74,000 / 月` 🏆** *(-91.2%)* | 🌟 **现代车企售后与大型经销商集团唯一标准** |

> 💡 **架构师一句话选型铁律**：**“汽车售后有 78% 的配件一年卖不到 3 件，绝不能用普通零售连续时间序列预测，更不能让 VLM 看着车头照片瞎猜零件号——必须采用每期自动衰减零销量概率的 `Teunter-Syntetos-Babai (TSB)` 间歇需求模型配合区域分拨中心（`PDC`）4 小时横向调拨池；通过单目 3D 凹陷深度 `VLM` + 历史拆解条件概率图谱提前预订保险杠背后的隐藏防撞梁与雷达支架；强制用 17 位 `VIN` 出厂配置表（`As-Built PR/RPO Codes`）与替换链 `AST` 锁死唯一原厂件号；且严禁任何未完成 100% 配件齐套入库扫码的事故车开上举升机！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大慢动件死库存、拆车暗伤补单与订错配件死穴）

### Q1：为什么 4S 店配件库里堆满了高达 `21.6%` 三年卖不出去的呆滞死库存，同时车间却天天因为缺件（`VOR 18.4%`）修不好车？`TSB` 模型 + 区域横向调拨怎么破？
- **大白话与第一性原理**：
  1. **经典移动平均甚至传统 Croston 模型的“零销量盲区”**：汽车售后 `145,000` 个 SKU 中，`78%` 是高度间歇性的慢动件（连续 40 周销量为 0，第 41 周突然因为一起追尾卖出 2 件）。传统移动平均在卖出后立刻补货；即使是经典 Croston 方法，也只在发生非零需求时才更新间隔估计——一旦某款老车型保有量萎缩进入长达两年的零销量期，Croston 的需求率估计值永远卡在高位不衰减，导致 180 家门店各自压货报废！
  2. **Tier 4 的“Teunter-Syntetos-Babai (`TSB`) 逐期概率指数衰减 + 区域多级共享池（Multi-Echelon Lateral Transshipment）”**：
     - 将需求拆解为**发生概率 $p_t$** 与**单次需求量 $z_t$**，在每一个零销量周期 $t$ 强制对发生概率做指数衰减：
       $$\hat{p}_t = \hat{p}_{t-1} + \beta \cdot \Big(\mathbb{I}(D_t > 0) - \hat{p}_{t-1}\Big), \quad \hat{z}_t = \hat{z}_{t-1} + \alpha \cdot (D_t - \hat{z}_{t-1}) \;\;(\text{仅当 } D_t > 0)$$
     - 凡是 $\hat{p}_t < 0.04$/周的超慢动高价件，**物理禁止 180 家单店各自备货**，统一收口在 4 座区域分拨仓（`PDC`）并通过同城 4 小时横向调拨（Lateral Transshipment `MILP`）共享安全库存——将缺件停驶率（`VOR`）从 `18.4%` 压至 **`0.6%`**，呆滞报废率从 `21.6%` 砍至 **`1.8%`**！

### Q2：为什么只靠 2D 外观照片定损（Tier 2 / Tier 3）到了车间拆开保险杠后，有 `26.5%` 的车要停工追加订件（Supplement）？怎么做到 `0.0%` 订错高低配件号？
- **单目 3D 凹陷能量变形场 `VLM` + 历史拆解条件概率图 + 17 位 `VIN` 出厂配置码（`PR / RPO`）铁闸**：
  1. **看穿塑料蒙皮背后的内部暗伤**：塑料保险杠具有回弹性，表面看似只有一道折痕，里面的铝合金防撞梁、吸能盒（Crush Cans）和盲区毫米波雷达支架早已变形断裂。Tier 4 用单目 3D 深度视觉模型测量覆盖件凹陷毫米级曲率，输入基于百万张历史拆解工单训练的**贝叶斯条件损坏图谱** $P(\text{内部件 } k \text{ 损坏} \mid \text{外板形变深度}, \text{碰撞角度})$，对概率 $\ge 0.85$ 的内部暗伤件在车主到店前就随首单同步预调拨（追加补单返工率从 `26.5%` 骤降至 **`1.9%`**）！
  2. **17 位 `VIN` 出厂配置单（As-Built Build Sheet）+ 零件号替换链（Supersession DAG）**：同款 2024 年 SUV 的右前大灯就有标准 LED、矩阵式激光大灯（`PR-8IT`）、带夜视清洗等 6 个不同原厂件号，外观照片几乎一模一样。Tier 4 强制通过 17 位 `VIN` 码反查主机厂出厂 `PR / RPO` 配置清单，并自动遍历原厂电子备件目录（`EPC`）中的**新旧件号替换链（如旧号 `A` 已停产替换为新号 `C` 且必须捆绑加订转接线束 `K`）**，实现 **`0.0%` 订错件逃逸**！

### Q3：如何防止事故车因为少一颗螺丝就把门店每天净赚 `$1,200` 的举升机工位堵死两天？
- **“配件齐套入库扫码（Parts-in-Bin Gate）”硬联锁工位与高压资质排程 `MILP`**：
  - 在门店 `DMS` 车间排程 `MILP` 中设立硬铁律：**对于非紧急安全件的预约维保与钣喷车辆，只有在系统确认该 `VIN` 工单名下 100% 的主件与暗伤预判件均已通过条码枪扫入本店备件格口（Bin Scan Confirmed）后，才允许分配举升机工位与技师工时**；
  - 同时严格匹配技师资质（如 800V 高压纯电车型电池包拆装必须绑定具备 `HV Level-3` 绝缘认证的高级技师与专用举升机），彻底消灭“车占着举升机等配件”的产能黑洞！

---

## ③ 🧭 直观可视化图解（决策树 + TSB/VIN-VLM/工位架构图 + 暗伤预判与齐套排产时序图）

### 3.1 企业级汽车售后备件、碰撞定损与车间排程架构选型决策树

```mermaid
flowchart TD
    A["企业级汽车售后备件网络、碰撞定损与 4S 车间调度<br/>(180 家门店 + 4 座区域仓 + 14.5 万 SKU + 日均 2,800 张工单)"] --> B{"是否仍用 DMS 移动平均补货 + 拆车后补单<br/>或让通用 VLM 凭照片直接猜零件号?"}
    B -- "是 (移动平均或裸 VLM 猜件号)" --> C["⛔ 缺件停驶、满仓死库存与拆车返工 (Tier 1 / Tier 2)<br/>18.4% 缺件停驶，21.6% 呆滞报废，26.5% 拆开才补订暗伤件，9.6% 订错高低配件"]
    B -- "引入需求预测与视觉分割" --> D{"是否采用 TSB 间歇需求多级池化、3D 暗伤概率图、<br/>17 位 VIN 出厂配置 AST 与配件齐套工位 MILP?"}
    D -- "否 (普通零售连续 ML + 2D 外观分割)" --> E["⚠️ 慢动件失准与内部暗伤漏判 (Tier 3)<br/>78% 慢动件预测失真，且看不见保险杠内部防撞梁/雷达损坏 (12.8% 补单)"]
    D -- "TSB 共享池 + VIN 暗伤 VLM + 齐套工位 MILP" --> F["🟩 Tier 4: TSB 多级共享池 + 17位 VIN 暗伤 VLM + 齐套工位 MILP 🏆<br/>(0.6% 缺件率 | 1.8% 死库存 | 1.9% 追加补单 | 0.0% 错件 | 省 91.2% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `TSB` 间歇需求多级池 + 17 位 `VIN` 暗伤 `VLM` + 齐套工位 `MILP` 架构图

```mermaid
flowchart LR
    subgraph Inventory["1. 14.5万 SKU TSB 间歇需求与区域池"]
        DMS["180家门店销量流<br/>(78% 为慢动间歇件)"] --> TSB["📦 TSB 概率衰减模型 (p_t * z_t)<br/>零销量期自动下调安全库存<br/>+ 4座区域 PDC 4h 横向调拨 MILP"]
    end

    subgraph Triage["2. 3D 暗伤 VLM 与 17位 VIN EPC 铁闸"]
        Photo["📸 车主碰撞照片 + 17位 VIN"] --> VLM["🔍 单目 3D 凹陷深度 + 拆解贝叶斯图<br/>预判内部防撞梁/雷达支架暗伤 (1.9% 补单)"]
        VLM --> EPC["🔒 主机厂 As-Built PR码 + 替换链 AST<br/>精确锁死原厂件号与捆绑线束 (0.0% 错件)"]
    end

    subgraph Bay["3. 配件齐套入格与高压资质工位 MILP"]
        TSB & EPC --> Bin["✅ 备件格口 100% 齐套扫码门禁"]
        Bin --> Lift["🔧 举升机 + 800V 高压技师 + 代步车 MILP<br/>(杜绝拆散车霸占工位等件)"]
    end
```

### 3.3 从“手机照片预判内部雷达暗伤与 17 位 VIN 锁件”到“齐套入库上举升机一次修竣”时序图

```mermaid
sequenceDiagram
    autonumber
    participant Owner as 事故车主 (#RO-7712, 2024 SUV)
    participant VLM as 3D 暗伤 VLM & 17位 VIN EPC 引擎
    participant PDC as TSB 区域备件池 & 横向调拨 MILP
    participant Bay as 门店备件格口 & 车间工位排程 MILP

    Owner->>VLM: 上传右后追尾 6 张照片 + 17 位 VIN 码（外观看似仅后保塑料蒙皮凹陷）
    VLM->>VLM: 1. 3D 深度测算凹陷曲率 -> 拆解条件概率图判定内部后防撞梁 (p=0.94) 与盲区雷达支架 (p=0.91) 已暗伤变形！
    VLM->>VLM: 2. 解码 17 位 VIN 出厂 PR 码 (PR-8IT 矩阵灯 + PR-7Y1 盲区监测) 并遍历替换链 -> 锁死精确原厂件号与转接线束 (0.0% 错件)
    VLM->>PDC: 触发区域 PDC 4 小时横向调拨班车，将外蒙皮 + 内部防撞梁 + 雷达支架一次性发往门店
    PDC-->>Bay: 周二 18:00 全部 5 件配件扫入门店 #14 备件格口 -> 触发“100% 齐套就绪”事件！
    Bay->>Owner: ⚡ 自动锁定周三 09:00 带 ADAS 标定架的 #14 工位与技师，进店当天一次性拆装标定交车（零拆检等待返工）！
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：DMS 移动平均 + 拆车补单 | 🟨 Tier 2：纯 VLM 看图猜件号 | 🟦 Tier 3：零售正态 ML + 2D 分割 | 🟩 Tier 4：`TSB` 多级池 + `VIN` 暗伤 `VLM` + 工位 `MILP` |
|---|---|---|---|---|
| **缺件停驶率 (`VOR Stockout`)** | `18.4%` ✕ | `15.8%` ✕ | `7.4%` ✕（慢动峰值漏判） | 🏆 **`0.6%`（`TSB` 间歇预测 + 区域 `PDC` 4h 横向调拨）** |
| **呆滞死库存报废率 (`Obsolescence`)** | `21.6%` ✕（偶发销量触发盲补） | `19.4%` ✕ | `11.2%` ✕（零销量期不衰减） | 🏆 **`1.8%`（`TSB` 概率逐期衰减 + 慢动件集中池化）** |
| **拆检追加补单与返工率 (`Supplement`)** | `26.5%` ✕（拆开才发现暗伤） | `31.2%` ✕ | `12.8%` ✕（仅看 2D 表面划痕） | 🏆 **`1.9%`（单目 3D 凹陷深度 + 拆解条件概率图预订）** |
| **订错 `VIN` 配件号装车逃逸率** | `4.2%` ✕（人工漏看改款替换号） | `9.6%` ✕（高低配外观混淆） | `1.9%` ✕ | 🏆 **`0.0%`（17 位 `VIN` 出厂 `PR/RPO` 码 + `EPC` 替换链 `AST`）** |
| **月度缺件、死库存与工位损失 (`TCO`)** | `$840,000 / 月` | `$950,000 / 月` | `$380,000 / 月` | **`$74,000 / 月` 🏆（节省 91.2%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（180 家 4S 门店 + 4 座区域仓 / 14.5 万个 SKU / 日均 2,800 张维修工单）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级汽车售后备件 TSB、VIN 碰撞 VLM 与工位排程定容计算器](../reports/enterprise-automotive-aftermarket-parts-croston-collision-vlm-bay-milp-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-automotive-aftermarket-parts-croston-collision-vlm-bay-milp-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection/run.py --check
  python -m unittest tests/test_enterprise_automotive_aftermarket_parts_croston_collision_vlm_bay_milp_selection.py
  ```

### 5.2 核心量化对比图（缺件停驶率 VOR Stockout % vs. 拆检追加补单返工率 Supplement Rework %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: DMS Moving-Avg + FIFO", "metric": "VOR Parts Stockout (%)", "pct": 18.4 },
      { "tier": "Tier 1: DMS Moving-Avg + FIFO", "metric": "Collision Supplement Rework (%)", "pct": 26.5 },
      { "tier": "Tier 2: Prompt Photo VLM Bot", "metric": "VOR Parts Stockout (%)", "pct": 15.8 },
      { "tier": "Tier 2: Prompt Photo VLM Bot", "metric": "Collision Supplement Rework (%)", "pct": 31.2 },
      { "tier": "Tier 3: Retail Gaussian ML + 2D", "metric": "VOR Parts Stockout (%)", "pct": 7.4 },
      { "tier": "Tier 3: Retail Gaussian ML + 2D", "metric": "Collision Supplement Rework (%)", "pct": 12.8 },
      { "tier": "Tier 4: TSB + VIN VLM + Bay MILP", "metric": "VOR Parts Stockout (%)", "pct": 0.6 },
      { "tier": "Tier 4: TSB + VIN VLM + Bay MILP", "metric": "Collision Supplement Rework (%)", "pct": 1.9 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 cuts VOR parts stockouts to 0.6% and collision supplement rework to 1.9%",
    "subtitle": "Comparison across 184 dealership/PDC sites and 145,000 OEM part SKUs (lower is better)",
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

- [ ] **对占比 70%+ 的慢动/间歇性汽车备件停用移动平均与普通正态时间序列，全面切换至 `Teunter-Syntetos-Babai (TSB)` 概率衰减模型**：在零销量周期持续衰减需求发生概率 $\hat{p}_t$，并将超慢动高价件上收至区域 `PDC` 共享池。
- [ ] **任何配件采购单下发前必须通过“17 位 `VIN` 出厂配置表（`As-Built PR/RPO Codes`）+ 原厂 `EPC` 零件替换链有向图”双重校验**：自动带上最新替换件号及强制捆绑的转接线束/卡扣包，严禁仅凭图片外观猜零件号。
- [ ] **碰撞照片定损必须结合“单目 3D 变形深度估计 + 历史拆解暗伤条件概率图谱”**：对覆盖件下方高概率受损的防撞梁、吸能盒与雷达支架进行首单预调拨。
- [ ] **车间工位排程 `MILP` 必须强制绑定“备件格口 100% 齐套入库扫码门禁（Parts-in-Bin Gate）”**：配件未到齐前严禁将非紧急事故车吊上举升机拆散占坑。
- [ ] **将区域极端天气预警（冰雹、暴雪黑冰）接入 `PDC` 前置铺货模型**：在寒潮冰雹过境前 48 小时向受影响门店预调拨挡风玻璃、前保蒙皮与大灯总成。
