# 企业级远洋航运（Maritime Container & Bulk Shipping）、气象洋流减阻航线规划（`IMO CII / EU ETS`）、港口拥堵准时到港（`JIT Virtual Arrival`）与滞期费/滞箱费（`BIMCO Laytime & D&D`）审计 Agent 架构选型指南：传统大圆航线全速抢锚+手工 Excel 算滞期费 vs. 纯 LLM 读租船合同 vs. 单纯海面气象导航 vs. 3D 波浪极曲线等时线 + `AIS` 泊位队列三次方功率降速 + `BIMCO Laytime` 区间状态机 Agent

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-maritime-shipping-weather-routing-port-demurrage-agent-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、远洋班轮/干散货航运公司船队运营副总裁（VP of Fleet Operations）、全球货代与跨国制造供应链物流总监（Global Logistics Director）及海事法务理算团队的生产落地指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection/results.json`](../../experiments/enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection/results.json)。

![企业级远洋航运气象导航与港口准时到达及滞期费审计 Agent 架构选型总结图](../../assets/svg/enterprise-maritime-shipping-weather-routing-port-demurrage-agent-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在管理 **65 艘远洋集装箱/干散货船队（年运量 `850,000 TEU`、每月 `280` 个挂靠港次、低硫重油 VLSFO 叠加欧盟 `EU ETS` 航运碳税综合成本 `$720/吨`）** 时，船东、租家（Charterers）与跨国货主面临三大“吞金巨兽”：
1. **“全速狂奔去锚地排队（Hurry-Up-and-Wait）”烧掉冤枉油与碳配额**：船舶推进功率与航速的三次方成正比（**海军部系数定律 $P \propto V^3$**）。传统调度（Tier 1）让一艘 14,000 TEU 巨轮顶着风浪以 `19.5 节` 全速赶往鹿特丹港，结果到港后因码头岸桥拥堵在锚地白白抛锚等了 **48 小时**——单航次多烧 **`410 吨` 重油与欧盟碳配额（白扔 `$295,200`）**，且导致 `IMO CII` 碳强度评级面临 **`31.0%` 的降级风险**！
2. **用通用大模型直接读租船合同（Charterparty）与装卸事实记录（`SOF`）算滞期费（Demurrage），多付 `29.0%` 冤枉钱**：海事滞期费理算涉及极其严苛的 `BIMCO` 法律状态转换（如 **`WWD SHEX UU` 晴天工作日周日节假日除外除非已使用**，以及一旦进入滞期则周末下雨照算不误的 **“一旦滞期永远滞期 Once on Demurrage, Always on Demurrage”** 铁律）。Tier 2 让通用 LLM 做文本心算，在跨越失效时点（Laytime Expiry）的雨停与吊机故障重叠区间上频频算错，导致租家多付 **`29.0%`** 的滞期费！
3. **进口重箱免箱期（Detention Free-Time）按先到先还（FIFO）调度导致天价阶梯罚金**：未结合船公司 `UIIIA` 免租期倒计时与灰池（Grey-Pool）街转箱匹配。

在 **65 艘运营船舶、每月 280 次挂靠与滞期理算、油+碳综合成本 `$720/吨`** 的企业级基准规模下，四档远洋航运与滞期理算架构的量化对决如下：

| 方案档位 | 核心架构组合 | 燃油与 `EU ETS` 碳成本多烧率 | 靠泊时间预测误差 (`ETA MAE`) | 滞期费 (`Demurrage`) 误付/多付率 | `IMO CII` 评级降级风险 | 月度多烧燃油与滞期滞箱总损耗 (`TCO`) | 生产推荐度 |
|---|---|---|---|---|---|---|
| **🟥 Tier 1：传统大圆航线全速抢锚 + 手工 Excel 算 SOF** | 按最短几何距离 19.5 节全速开船去锚地排队 + 人工用 Excel 核对装卸时间表 | `18.2%` ✕ | `32.0 小时` ✕ | `24.5%` ✕ | `31.0%` ✕ | `$780,000 / 月` ✕ | ⛔ 三次方高航速白烧油去锚地干等 |
| **🟨 Tier 2：纯大模型读租船合同与港口新闻 (`Prompt LLM`)** | 把 BIMCO 合同与 SOF 丢给通用 LLM 心算扣减时间并推荐航速 | `16.5%` ✕ | `28.5 小时` ✕ | `29.0%` ✕ *(漏算一旦滞期铁律)* | `27.5%` ✕ | `$820,000 / 月` ✕ | ⛔ 缺乏流体力学与法律区间状态机 |
| **🟦 Tier 3：单纯开阔海面气象导航 (无港口队列与法务 AST)** | 仅避开台风但未打通目的港 AIS 岸桥排队队列，无滞期费区间代数引擎 | `8.4%` | `11.5 小时` | `14.2%` ✕ | `11.0%` | `$365,000 / 月` ✕ | ⚠️ 避开了风浪，却仍在港口锚地排队 |
| **🟩 Tier 4：3D 波浪极曲线等时线 + `AIS` 港口队列 `JIT` 降速 + `BIMCO Laytime` 状态机 Agent** | **3D 海浪/洋流 `Kwon` 失速极曲线等时线 + 目的港 `AIS` 岸桥队列 `V^3 JIT` 虚拟到达降速 + `IMO CII` 碳边界 + `BIMCO` 滞期区间代数 `FSM`** | **`1.2%` 🏆** *(-17.0 pts)* | **`1.8 小时` 🏆** *(准点靠泊)* | **`0.0%` 🏆** *(分毫不差)* | **`0.0%` 🏆** *(保 A/B 级)* | **`$68,000 / 月` 🏆** *(-91.3%)* | 🌟 **现代航运船队与全球货主唯一生产架构** |

> 💡 **架构师一句话选型铁律**：**“远洋航运省油降碳的核心在‘用目的港泊位空闲时刻倒推经济航速’——必须将目的港 `AIS` 岸桥排队预测与船舶 `P ∝ V^3` 三次方推进定律及 3D 海浪失速极曲线耦合，通过 `BIMCO Virtual Arrival（虚拟到达）` 条款把航速从 19.5 节精准降到 15.2 节准点直靠；而在滞期费（Demurrage）理算上，必须把 `SOF` 事实记录编译为确定性的‘时间区间并集与一旦滞期状态机（FSM）’，彻底终结大模型心算冤枉账！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大航运燃油、碳税与滞期理算痛点）

### Q1：为什么船长沿着地图上最短的“大圆航线（Great-Circle Route）”全速开船反而多烧了 `18.2%` 的油？3D 波浪极曲线 + `AIS` 准时到达（`JIT`）怎么单航次省下 `$29.5 万`？
- **大白话与第一性原理**：
  1. **三次方推进定律（Cubic Law $P \propto V^3$）的恐怖弹性**：船舶克服水阻力的功率与速度的三次方成正比——**航速只要降低 `20%`（如从 `19.5 节` 降到 `15.6 节`），每日燃油消耗直接暴降近 `49%`**！如果目的港鹿特丹因为前面有 6 条船积压，泊位要 48 小时后才空出来，早到 48 小时唯一的结果就是在锚地白烧 `$295,200`（`410 吨` VLSFO + 欧盟 `EU ETS` 碳配额）！
  2. **几何最短 $\neq$ 阻力最小（顶浪失速与参数横摇 Parametric Roll）**：北太平洋冬季大圆航线正好穿过有效波高 $H_s = 7.5\text{m}$ 的顶浪风暴区，船体附加波浪阻力令每海里油耗暴增 `+38%`，甚至诱发集装箱成排倒塌的参数横摇共振。
  3. **Tier 4 的“海面 3D 等时线 + 港口 `AIS` 岸桥队列反推航速”双闭环**：
     - 航线层：基于 `ECMWF` 海浪方向谱与船体特定的 **`Kwon` 波浪失速极曲线（Speed-Loss Polar Curves）** 运行 3D 动态等时线搜索，向南微绕 180 海里顺借黑潮（Kuroshio）洋流并避开参数横摇区；
     - 泊位层：实时追踪目的港锚地与泊位内所有船舶的 `AIS` 吃水变化与岸桥作业速率，提前 5 天预测精确靠泊窗口（`ETA MAE` 仅 **`1.8 小时`**），自动激活 **`BIMCO Virtual Arrival` 虚拟到达协议**将航速平滑降至 `15.2 节`，单航次立省 `410 吨` 燃油且将 `IMO CII` 评级稳稳锁定在 `A/B` 级！

### Q2：在审核一笔 `$185,000` 的滞期费（Demurrage）账单时，为什么通用大模型（Tier 2）会算错 `$64,200`？`BIMCO Laytime` 区间代数状态机是怎么算的？
- **海事法务铁律与状态机跳变**：
  - 租船合同约定许可装卸时间（Allowed Laytime）为 `96 小时`，条款为 **`WWD SHEX UU`（晴天工作日，周日与节假日除外，除非实际使用）**。
  - **两阶段状态跳变点（Expiry Boundary $t_{\text{exp}}$）**：
    1. **在许可装卸时间届满前（$t < t_{\text{exp}}$，On Laytime）**：周二下暴雨停工的 14 小时、岸桥机械故障的 6 小时（若同时重叠发生，必须按**布尔区间并集 Interval Union** 去重，不能重复扣减！）以及周日未开工时间，**统统不计入已用装卸时间**，从而将真正的届满时刻 $t_{\text{exp}}$ 向后顺延！
    2. **一旦越过真实届满时刻（$t \ge t_{\text{exp}}$，Once on Demurrage, Always on Demurrage）**：除非合同另有极其罕见的明确排除，否则此后的周日、节假日乃至暴雨停工时间**全部 100% 连续计收滞期费**！
  - Tier 2 的通用大模型既不会做重叠时间段的区间并集去重，又把周二大雨扣减错误地漏算或与届满后的周日混淆。Tier 4 用多模态抽取器将船长签字的《装卸事实记录（`SOF`）》解析为标准化时间轴事件，交由确定性 **`BIMCO Laytime` 有限状态机** 逐分钟推进求解，当场有理有据核减 **`$64,200`** 虚高索赔，实现 **`0.0%` 滞期费误付**！

### Q3：面对到港后的 4,200 个 40 尺进口重箱，如何消灭高达 `$165/天/箱` 的集装箱超期滞箱费（Detention）？
- **`UIIIA` 免箱期倒计时 + 灰池街转箱（Street-Turn）最小费用流规划**：
  - 传统车队按先到先派（FIFO）提还箱（Tier 1），结果把还有 6 天免租期的箱子早早还进堆场，却让另外 380 个箱子拖过第 10 天免箱期红线触发阶梯式天价罚金。Tier 4 将每个箱号的船公司免箱期倒计时、出口工厂就近用箱需求（直接原箱“街转 Street-Turn”免去一进一出空箱吊装费）与场外堆场预约配额建模为**最小费用网络流（Min-Cost Network Flow）**，提前 72 小时优先清零临期高危箱！

---

## ③ 🧭 直观可视化图解（决策树 + 气象导航与滞期理算架构图 + 航次协同流程图）

### 3.1 企业级远洋航运气象导航、港口 JIT 与滞期费审计架构决策树

```mermaid
flowchart TD
    A["企业级远洋航运船队调度与滞期滞箱 (D&D) 管理<br/>(65 艘船 + 85万 TEU/年 + 每月 280 次挂靠理算)"] --> B{"是否仍按大圆航线全速抢锚排队<br/>或用纯 LLM 心算 SOF 滞期费账单?"}
    B -- "是 (全速抢锚或 LLM 心算账单)" --> C["⛔ 燃油与赔款双失血 (Tier 1 / Tier 2)<br/>18.2% 燃油碳税多烧，且滞期费账单多付 24.5%~29.0%"]
    B -- "引入海面气象导航" --> D{"是否打通了目的港 AIS 岸桥排队 JIT 降速、<br/>IMO CII 碳边界与 BIMCO 滞期区间状态机?"}
    D -- "否 (仅开阔海面避风浪)" --> E["⚠️ 锚地空等与理算漏扣 (Tier 3)<br/>靠泊预测误差仍达 11.5h，且无法处理重叠 SOF 停工与一旦滞期规则"]
    D -- "3D 极曲线等时线 + AIS JIT + BIMCO 状态机" --> F["🟩 Tier 4: 3D 波浪等时线 + AIS 港口 JIT 降速 + BIMCO 滞期状态机 🏆<br/>(1.2% 燃油超耗 | 1.8h 靠泊误差 | 0.0% 滞期多付 | 节省 91.3% 损耗)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 3D 波浪等时线 + `AIS` 泊位队列 `JIT` + `BIMCO Laytime` 状态机架构图

```mermaid
flowchart LR
    subgraph Ocean["1. 3D 气象洋流与碳强度导航层"]
        Met["🌊 ECMWF 海浪谱 + 洋流场<br/>+ 船体 Kwon 失速极曲线"] --> Iso["🧭 3D 动态等时线求解器<br/>规避参数横摇 & 锁定 IMO CII A/B 级"]
    end

    subgraph Port["2. 目的港 AIS 队列与三次方 JIT 降速"]
        AIS["⚓ 目的港 AIS 吃水/岸桥作业流"] --> Queue["⏱️ 泊位空闲窗口图预测<br/>(1.8h ETA MAE)"]
        Iso --> JIT["⚡ P ∝ V^3 虚拟到达 (Virtual Arrival)<br/>19.5kts -> 15.2kts 省 410 吨燃油"]
        Queue --> JIT
    end

    subgraph Legal["3. BIMCO 滞期状态机与集装箱流"]
        SOF["📄 船长签字 SOF 事实记录 + 租约"] --> FSM["🛡️ BIMCO Laytime 区间代数状态机<br/>WWD SHEX UU 去重扣减 & 一旦滞期铁律<br/>(0.0% 滞期费多付)"]
        Boxes["📦 4,200 个进口集装箱免租期"] --> Flow["🚚 UIIIA 街转箱最小费用流调度"]
    end
```

### 3.3 从“鹿特丹港前 48 小时拥堵预警”到“单航次省 410 吨燃油 + 核减 $64,200 滞期费”时序图

```mermaid
sequenceDiagram
    autonumber
    participant Ship as 14,000-TEU 集装箱船 (#VOY-NLRTM-42)
    participant JIT as 3D 波浪等时线 + AIS 泊位队列引擎
    participant Charter as 船东 & 租家 (Virtual Arrival 协议)
    participant FSM as BIMCO Laytime 区间代数状态机

    JIT->>JIT: 距离鹿特丹 1,400 海里时，AIS 岸桥队列模型检出目的港泊位积压 48 小时 (ETA MAE 1.8h)
    JIT->>Charter: 自动生成 BIMCO Virtual Arrival 降速方案：航速由 19.5 节降至 15.2 节，靠泊时刻零延误
    Charter->>Ship: ⚡ 确认下发 15.2 节经济航速指令 (Act automatically) 并微调航向顺黑潮/洋流避开 7.5m 顶浪
    Note over Ship,Charter: 准点直靠泊位，单航次节省 410 吨 VLSFO 与 EU ETS 碳配额 ($295,200)，IMO CII 保持 A 级！
    Charter->>FSM: 完货后船东提交 $185,000 滞期费索赔单与 12 页 SOF 装卸事实记录
    FSM->>FSM: 执行区间并集去重：核减届满前周二暴雨 14h + 岸桥故障净非重叠 4h + 周日未用时段，精确重定届满点 t_exp
    FSM-->>Charter: ✅ 输出可审计理算书 (Confirm)：合法滞期费实为 $120,800，当场驳回 $64,200 虚高索赔 (0.0% 误付)！
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：大圆航线全速 + Excel | 🟨 Tier 2：纯 LLM 读合同与新闻 | 🟦 Tier 3：单纯海面气象导航 | 🟩 Tier 4：3D 等时线 + 港口 JIT + BIMCO FSM |
|---|---|---|---|---|
| **燃油与 `EU ETS` 碳配额多烧率** | `18.2%` ✕（19.5 节全速去锚地干等） | `16.5%` ✕ | `8.4%`（未与港口岸桥队列联动） | 🏆 **`1.2%`（`P ∝ V^3` 虚拟到达降速 + 洋流助航）** |
| **目的港实际靠泊预测误差 (`ETA MAE`)** | `32.0 小时` ✕ | `28.5 小时` ✕ | `11.5 小时` | 🏆 **`1.8 小时`（融合港口 `AIS` 吃水与岸桥作业图）** |
| **滞期费 (`Demurrage`) 误付/多付率** | `24.5%` ✕（人工漏扣重叠停工） | `29.0%` ✕（LLM 搞错一旦滞期边界） | `14.2%` ✕ | 🏆 **`0.0%`（`BIMCO Laytime` 区间代数确定性状态机）** |
| **`IMO CII` 年度碳强度降级风险** | `31.0%` ✕（沦为 D/E 级面临限航） | `27.5%` ✕ | `11.0%` | 🏆 **`0.0%`（航次级 CII 碳排放包络线硬约束）** |
| **65 艘船队月度多烧油与滞期总损耗 (`TCO`)** | `$780,000 / 月` | `$820,000 / 月` | `$365,000 / 月` | **`$68,000 / 月` 🏆（节省 91.3%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（65 艘远洋船舶 / 85 万 TEU / 月均 280 次挂靠理算）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级远洋航运气象导航与港口 JIT 及滞期理算定容计算器](../reports/enterprise-maritime-shipping-weather-routing-port-demurrage-agent-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-maritime-shipping-weather-routing-port-demurrage-agent-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_maritime_shipping_weather_routing_port_demurrage_agent_selection.py
  ```

### 5.2 核心量化对比图（燃油与碳配额多烧率 % vs. 滞期费账单多付率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Great-Circle + Excel", "metric": "Bunker/ETS Overburn (%)", "pct": 18.2 },
      { "tier": "Tier 1: Great-Circle + Excel", "metric": "Demurrage Overpayment (%)", "pct": 24.5 },
      { "tier": "Tier 2: Prompt Charterparty LLM", "metric": "Bunker/ETS Overburn (%)", "pct": 16.5 },
      { "tier": "Tier 2: Prompt Charterparty LLM", "metric": "Demurrage Overpayment (%)", "pct": 29.0 },
      { "tier": "Tier 3: Open-Sea Router Only", "metric": "Bunker/ETS Overburn (%)", "pct": 8.4 },
      { "tier": "Tier 3: Open-Sea Router Only", "metric": "Demurrage Overpayment (%)", "pct": 14.2 },
      { "tier": "Tier 4: 3D Isochrone + Port JIT + FSM", "metric": "Bunker/ETS Overburn (%)", "pct": 1.2 },
      { "tier": "Tier 4: 3D Isochrone + Port JIT + FSM", "metric": "Demurrage Overpayment (%)", "pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 cuts bunker/carbon overburn to 1.2% and eliminates demurrage overpayments (0.0%)",
    "subtitle": "Comparison across a 65-vessel fleet and 280 monthly port calls (lower is better)",
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

- [ ] **打通开阔海面气象导航与目的港 `AIS` 泊位/岸桥排队队列预测（`Just-in-Time Virtual Arrival`）**：一旦检出目的港候泊延迟 `>6 小时`，立即按三次方推进定律 $P \propto V^3$ 下调经济航速，杜绝全速赶去锚地抛锚。
- [ ] **航线规划必须采用船体专属的 `Kwon` 波浪失速极曲线与参数横摇（Parametric Roll）避险包络**：禁止使用不考虑波浪入射角与船体吃水状态的静态最短几何路径。
- [ ] **将 `IMO CII` 年度碳强度轨迹与欧盟 `EU ETS` 碳价直接嵌入航次目标函数**：确保每条船舶全年稳定维持在 `CII A/B` 评级安全区内。
- [ ] **滞期费（Demurrage）理算强制使用确定性 `BIMCO Laytime` 区间代数状态机（`FSM`）**：LLM 仅负责将纸质/PDF `SOF` 抽取为结构化时间段，所有重叠停工去重（Interval Union）与 `Once on Demurrage` 届满翻转必须由确定性代码执行。
- [ ] **集装箱陆运提还箱接入免租期倒计时（Free-Time Clock）与“街转箱（Street-Turn）”最小费用流求解器**：优先调度临近免箱期红线的重箱与空箱。
