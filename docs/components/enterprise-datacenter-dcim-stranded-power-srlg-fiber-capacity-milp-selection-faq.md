# 企业级数据中心与智算机房容量规划（`DCIM`）、五维搁浅电力回收（`5D Capacity-Tetris MILP`）、三相 `PDU` 相位均衡（`L1/L2/L3`）与光纤跳线共享风险链路组（`SRLG Suurballe`）物理隔离 Agent 架构选型指南：传统铭牌功率 Excel+就近插线 vs. 纯 LLM 排机柜 U 位与跳线 vs. 基础 DCIM 资产库 vs. `P99` 遥测五维俄罗斯方块 `MILP` + 三相 `AB/BC/CA` 倒闸配平 + `Suurballe SRLG` 不相交光路与机柜竣工 `VLM` Agent

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-datacenter-dcim-stranded-power-srlg-fiber-capacity-milp-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、大型智算中心与企业数据中心基础设施总监（Head of Data Center Operations & Capacity Planning）、网络布线架构师及 `DCIM`（`NetBox / Sunbird / Nlyte / 施耐德 EcoStruxure`）平台负责人的生产落地指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_datacenter_dcim_stranded_power_srlg_fiber_capacity_milp_selection/results.json`](../../experiments/enterprise_datacenter_dcim_stranded_power_srlg_fiber_capacity_milp_selection/results.json)。

![企业级数据中心 DCIM 五维搁浅电力回收、三相 PDU 配平与 SRLG 光纤隔离 Agent 架构总结图](../../assets/svg/enterprise-datacenter-dcim-stranded-power-srlg-fiber-capacity-milp-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在运营一座 **总 IT 供电容量 `40 MW`、管理 `1,600` 个高密与混部机柜、维护 `48,000` 条光纤与高速铜缆跳线（Cross-Connects，每搁浅 1 MW 电力相当于白白冻结 `$1,000 万` 机房基建 CapEx）** 的企业数据中心或智算中心时，基础设施团队普遍面临三大“明明有资源却装不下、装上去又跳闸断网”的顽疾：
1. **按服务器双电源铭牌功率（Nameplate Wattage）在 Excel 里死板扣减配额 + 贪心就近上架（Tier 1），造成高达 `31.5%（12.6 MW）` 的“碎片化搁浅电力（Stranded Power）”与 `11.2%` 的 `3米 DAC` 铜缆超距返工**：机柜 A 剩 15U 空间但供电满了，机柜 B 剩 8kW 电但冷板液冷（`CDU`）流量满了，或者把同一计算集群的 64 台 GPU 服务器分散塞进相距 8 米的机柜，现场布线时才发现 **3.0 米无源高速铜缆（Passive DAC）根本够不着核心 Leaf 交换机**！
2. **机柜内 415V 三相智能 `PDU`（`L1-L2 / L2-L3 / L3-L1`）靠驻场技师顺手全插在底部插座组（Tier 1），导致单相严重偏载，一旦 A 路市电切换至 B 路（Failover）瞬间触发断路器过载跳闸（风险高达 `14.6%`）**！
3. **主备双路（A/B Redundant）400G 光纤跳线只按桥架最短路径敷设（Tier 1 / Tier 3），导致 `18.4%` 的“双路冗余光纤”其实偷偷穿过了同一个垂直弱电竖井或同一根地埋套管（违反 `SRLG` 共享风险链路组隔离），施工电钻一铲子下去主备全断**！

在 **40 MW 智算/数据中心、1,600 个机柜、48,000 条跳线** 的企业基准规模下，四档数据中心容量规划、配电与布线架构的量化对决如下：

| 方案档位 | 核心架构组合 | 碎片化搁浅不可用电力占比 (`Stranded MW`) | 主备光纤共管井违例率 (`SRLG Violation`) | 三相 `PDU` 偏载跳闸风险率 | `DAC/AOC` 线缆超距与错接返工率 | 月度搁浅基建折旧与布线返工总损失 (`TCO`) | 生产推荐度 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：铭牌功率 Excel 台账 + 就近塞机柜 + 最短路跳线** | 按 100% 电源铭牌锁额度 + 有空 U 位就塞 + 走最近桥架穿纤 | `31.5% (12.6 MW)` ✕ | `18.4%` ✕ *(主备同穿一根竖井)* | `14.6%` ✕ *(全插AB相倒闸必跳)* | `11.2%` ✕ *(超3米DAC物理极限)* | `$1,150,000 / 月` ✕ | ⛔ 冻结上亿基建电力，且单点故障频发 |
| **🟨 Tier 2：纯大模型排机柜 U 位与跳线表 (`Prompt LLM`)** | 把机房平面图与设备表喂给通用 LLM 排上架单与跳线路由 | `28.0% (11.2 MW)` ✕ | `24.5%` ✕ | `19.2%` ✕ *(无三相电流矢量概念)* | `16.8%` ✕ *(漏算垂直理线余量)* | `$1,280,000 / 月` ✕ | ⛔ 空间三维走线距离算错，施工全作废 |
| **🟦 Tier 3：标准 `DCIM` 资产库 + SNMP 峰值监控 (无五维 MILP)** | NetBox/Sunbird 记录 U 位与总功率，但无五维打包与 `Suurballe` 算法 | `14.2% (5.7 MW)` ✕ | `6.8%` ✕ | `5.4%` ✕ | `4.1%` ✕ | `$480,000 / 月` ✕ | ⚠️ 只能事后看监控，无法事前全局寻优 |
| **🟩 Tier 4：五维俄罗斯方块 `MILP` + 三相 `PDU` 配平 + `Suurballe SRLG` + 机柜 `VLM` Agent** | **`P99` 遥测五维打包 (`电/液冷/U高/承重/3m铜缆`) `MILP` + 三相 `AB/BC/CA` 倒闸配平 + `Suurballe` 物理不相交光路 + 竣工 `VLM`** | **`2.1% (0.8 MW)` 🏆** *(释放11.8MW)* | **`0.0%` 🏆** *(零共井共管)* | **`0.0%` 🏆** *(零偏载跳闸)* | **`0.0%` 🏆** *(零超距返工)* | **`$92,000 / 月` 🏆** *(-92.0%)* | 🌟 **现代高密 AI 智算中心与主机托管唯一标准** |

> 💡 **架构师一句话选型铁律**：**“高密智算中心上架决不能看哪儿有空 U 位就往哪儿塞——必须用‘五维容量俄罗斯方块 `MILP`（同时对齐 `P99 实测电功率 + CDU 液冷散热量 + 42U 高度 + 地板 1,800kg 承重 + 3.0 米无源铜缆三维走线半径`）’将搁浅电力从 31.5% 压至 2.1%；在机柜微粒度上用整数规划将服务器电源线均匀交错分配至三相 `PDU` 的 `AB / BC / CA` 插座排（确保单路掉电切换时每相电流 `<80% NEC` 脱扣红线）；并用图论 `Suurballe` 不相交最短路算法强制主备光纤在桥架、竖井与管道上 100% 物理零重叠！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大千万美元电力搁浅、三相偏载跳闸与主备光纤同断死穴）

### Q1：为什么一座 40 MW 的数据中心明明只用了 27 MW，机房经理却说“彻底满载、一台新 GPU 服务器都装不下了”？Tier 4 怎么凭空找回 `11.8 MW`（价值 `$1.18 亿`）可用容量？
- **大白话与第一性原理**：
  1. **水桶短板碎片化（Multi-Dimensional Resource Stranding）与铭牌虚胖**：
     - 一台 8 卡 H200 服务器配备了 `6 × 3,000W` 冗余电源模块（铭牌总和 `18 kW`），但实际满载 `P99` 功耗为 `10.2 kW`。Tier 1 按铭牌在 Excel 里死抠配额，直接虚占了 `40%` 假负载；
     - 更致命的是**五维错位**：贪心排柜（First-Fit）把吃电不占地方的 GPU 节点和占地方不吃电的存储节点乱塞一气，导致前排机柜“电满了但空着 25U”，后排机柜“U 位塞满了但剩下一半电”，且跨机柜距离超过 **3.0 米 800G 无源铜缆（DAC）** 极限（被迫加买昂贵且高功耗的 AOC 有源光模块）！
  2. **Tier 4 的“五维容量俄罗斯方块（5D Capacity-Tetris `MILP`）”**：
     - 将每一台待上架设备表示为五维向量 $\vec{d}_i = (\text{kW}_{P99}, \text{kW}_{\text{CDU}}, \text{U}, \text{kg}, \text{Rail\_Group})$，并在三维机房网格上联合约束：
       $$\text{Cable\_Length}_{3\text{D}}(r_i, r_{\text{leaf}}) = |x_i - x_s| + |y_i - y_s| + (42 - u_i + 42 - u_s)\cdot 0.0445\text{m} + 0.6\text{m}_{\text{slack}} \le 3.0\text{m}$$
     - 自动将冷热/轻重设备互补拼箱，把搁浅电力从 `31.5%（12.6 MW）` 压至 **`2.1%（0.8 MW）`**，直接省下新建一座 12 MW 变电站的巨额投资！

### Q2：机柜总功率明明没超标，为什么进行 A/B 路 UPS 例行切换演练时，整排机柜的 60A 三相 `PDU` 空气开关会突然啪地跳闸？
- **三相交流电（`L1-L2 / L2-L3 / L3-L1`）插座排偏载陷阱**：
  - 高密机柜广泛采用 415V/240V 60A 三相智能 `PDU`，内部将 36 个插座分为 **`Bank 1 (Phase AB)`、`Bank 2 (Phase BC)`、`Bank 3 (Phase CA)`** 三组独立过流保护支路。
  - 外包驻场工人在上架 6 台高功耗 GPU 服务器时，为了理线省事，把所有电源线全插在了最顺手的下排 `Phase AB` 插座组上！平时 A、B 双路供电各分担 50% 时相安无事；一旦 A 路停电检修、100% 负载瞬间切到 B 路 `PDU` 时，`Phase AB` 单相电流瞬间飙到 **`108%`** 当场跳闸，而旁边的 `Phase BC` 和 `Phase CA` 却空空如也！
  - Tier 4 在自动生成的施工跳线单（Cut-Sheet）中精确到**每一个具体插座孔号（如 `PDU-A Outlet #14 [Phase BC]`）**，强制满足：
    $$\forall p \in \{\text{AB}, \text{BC}, \text{CA}\}: \quad I_{\text{Feed\_A}}(p) + I_{\text{Feed\_B}}(p) \le 0.80 \times I_{\text{Breaker\_Rated}}$$
    将三相偏载与倒闸跳闸风险从 `14.6%` 彻底降为 **`0.0%`**！

### Q3：客户花了双倍价钱买“主备双路物理隔离光纤（Dual-Path Cross-Connect）”，为什么施工队一钻头下去还是主备同时断网？`Suurballe SRLG` 算法怎么防？
- **共享风险链路组（`Shared-Risk Link Group, SRLG`）边/管道不相交路由 + 机柜竣工照片 `VLM` 核验**：
  1. **告别两次独立 Dijkstra 最短路**：在机房桥架图上跑两次最短路，找出来的 A 路和 B 路往往在中间某一段共用同一个过道桥架、同一个防火穿墙孔或同一个垂直弱电竖井（`#R3`）。Tier 4 在 `NetBox / DCIM` 三维管道图上运行 **Suurballe 边与风险组完全不相交最短路算法（SRLG-Disjoint Shortest Path Pair）**，同时校核光功率衰减预算（$\sum \alpha_{\text{splice}} + \alpha_{\text{conn}} + L_{\text{km}}\cdot 0.35\text{ dB} \le 1.8\text{ dB}$），确保 $\text{SRLG}(P_A) \cap \text{SRLG}(P_B) \equiv \emptyset$（**`0.0%` 共管井违例**）！
  2. **通电前手机全景 `VLM` 竣工门禁**：驻场人员插完线后用手机拍一张机柜前后门照片，**机柜 `VLM`** 自动读取线缆二维码吊牌、交换机端口位置与 `PDU` 相位孔位，与 `DCIM` 电子施工单 100% 吻合后才解锁远程上电（**`0.0%` 错接返工**）！

---

## ③ 🧭 直观可视化图解（决策树 + 五维排柜/三相/SRLG 架构图 + 极速上架核验时序图）

### 3.1 企业级数据中心 DCIM 容量规划、配电与光纤路由选型决策树

```mermaid
flowchart TD
    A["企业级数据中心与智算机房 DCIM 容量规划<br/>(40 MW 总供电 + 1,600 个机柜 + 4.8 万条光纤/铜缆跳线)"] --> B{"是否仍按 100% 电源铭牌功率用 Excel 扣额度<br/>且任由工人就近插 PDU 与走最近桥架穿纤?"}
    B -- "是 (Excel 铭牌台账 + 人工随意插线)" --> C["⛔ 31.5% 电力搁浅浪费与倒闸跳闸 (Tier 1 / Tier 2)<br/>冻结 12.6 MW 宝贵电力，14.6% 三相偏载跳闸，18.4% 主备光纤同穿一井"]
    B -- "部署标准 DCIM 资产库" --> D{"是否具备 P99 五维俄罗斯方块 MILP、三相 PDU 插座级配平<br/>与 Suurballe SRLG 物理不相交光路规划?"}
    D -- "否 (仅用 DCIM 做资产登记与峰值监控)" --> E["⚠️ 跨维度碎片搁浅与隐性共管风险 (Tier 3)<br/>电/冷/U/铜缆半径未联合求解，仍有 14.2% 电力搁浅与 6.8% SRLG 重叠"]
    D -- "五维 MILP + 三相配平 + Suurballe SRLG + 竣工 VLM" --> F["🟩 Tier 4: 五维容量 MILP + 三相 PDU 配平 + SRLG 光路 Agent 🏆<br/>(2.1% 搁浅率 | 0.0% 偏载跳闸 | 0.0% 主备共井 | 省 92.0% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 五维容量俄罗斯方块 `MILP` + 三相 `PDU` 配平 + `Suurballe SRLG` 架构图

```mermaid
flowchart LR
    subgraph Tetris["1. P99 五维容量俄罗斯方块 MILP"]
        Telemetry["SNMP/Redfish P99 功耗<br/>+ CDU 液冷 + 42U + 地板承重"] --> MILP["🧱 五维机柜拼箱与 3m DAC 半径 MILP<br/>回收 11.8 MW 搁浅电力 (搁浅仅 2.1%)<br/>100% 满足 <=3.0m 无源铜缆距离"]
    end

    subgraph Power["2. 三相 PDU 插座级倒闸配平器"]
        MILP --> Phase["⚡ L1-L2 / L2-L3 / L3-L1 整数配平<br/>精确指定每个电源插孔 (Phase AB/BC/CA)<br/>锁死 A+B 倒闸电流 <= 80% NEC (0.0% 跳闸)"]
    end

    subgraph Fiber["3. Suurballe SRLG 不相交光路与竣工 VLM"]
        MILP --> Suurballe["🌐 Suurballe 桥架/竖井不相交路由<br/>SRLG(Path A) ∩ SRLG(Path B) == ∅<br/>+ 机柜二维码照片 VLM 竣工门禁"]
        Phase --> Suurballe
    end
```

### 3.3 从“64 台 H200 节点五维寻优排柜”到“三相孔位锁定与竣工照片 VLM 核验”时序图

```mermaid
sequenceDiagram
    autonumber
    participant Ops as 智算中心容量规划师
    participant MILP as 五维容量 MILP & 三相 PDU 求解器
    participant Fiber as Suurballe SRLG 光纤路由器
    participant VLM as 驻场施工 App & 机柜竣工 VLM

    Ops->>MILP: 申请上架 64 台 8xH200 节点（P99 实测 10.2 kW/台，要求与 Leaf 交换机距离 <=3.0m 无源铜缆）
    MILP->>MILP: 1. 求解五维拼箱 MILP：在 Row 4 连续机柜内紧凑排布，三维走线含垂直理线余量 <=2.82m (省去昂贵 AOC 光模块!)
    MILP->>MILP: 2. 生成精确插座施工单：将每台机器双电源交错分配至 PDU Bank AB / BC / CA（三相偏差 1.4%，倒闸负载 76% <= 80% 红线）
    MILP->>Fiber: 规划 Cage #12 至 MMR-1/MMR-2 的双路 400G 上联光路
    Fiber-->>VLM: Suurballe 算法输出完全不相交路径：A 路走西侧 #R1 竖井，B 路走东侧 #R4 竖井（0 共享管道，光损 1.42 dB）
    VLM->>VLM: 驻场技师插线完毕拍摄机柜背面全景照 -> VLM 识别全部 128 根网线 QR 码与 PDU 孔位 100% 匹配施工单，解锁上电！
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：铭牌 Excel + 就近插线 | 🟨 Tier 2：纯 LLM 排机柜与跳线 | 🟦 Tier 3：标准 DCIM + 峰值监控 | 🟩 Tier 4：五维 `MILP` + 三相配平 + `SRLG` |
|---|---|---|---|---|
| **搁浅不可用电力占比 (`Stranded MW`)** | `31.5% (12.6 MW)` ✕（铭牌虚占+碎片） | `28.0% (11.2 MW)` ✕ | `14.2% (5.7 MW)` ✕ | 🏆 **`2.1% (0.8 MW)`（释放 `11.8 MW` 宝贵供电容量）** |
| **主备光纤共管井违例率 (`SRLG`)** | `18.4%` ✕（最短路同穿一根竖井） | `24.5%` ✕ | `6.8%` ✕ | 🏆 **`0.0%`（`Suurballe` 桥架/管道完全不相交对算法）** |
| **三相 `PDU` 偏载跳闸风险率** | `14.6%` ✕（扎堆插下排 AB 相） | `19.2%` ✕ | `5.4%` ✕ | 🏆 **`0.0%`（`AB/BC/CA` 插座孔位级倒闸 `<80%` 配平）** |
| **`DAC/AOC` 超距与错接返工率** | `11.2%` ✕（超 3m 无源铜缆极限） | `16.8%` ✕ | `4.1%` ✕ | 🏆 **`0.0%`（三维走线距离约束 + 机柜照片 `VLM` 门禁）** |
| **月度搁浅基建折旧与返工总损失** | `$1,150,000 / 月` | `$1,280,000 / 月` | `$480,000 / 月` | **`$92,000 / 月` 🏆（节省 92.0%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（40 MW 数据中心 / 1,600 个机柜 / 48,000 条光纤与高速铜缆跳线）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级数据中心 DCIM 五维搁浅电力与 SRLG 光纤定容计算器](../reports/enterprise-datacenter-dcim-stranded-power-srlg-fiber-capacity-milp-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-datacenter-dcim-stranded-power-srlg-fiber-capacity-milp-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_datacenter_dcim_stranded_power_srlg_fiber_capacity_milp_selection/run.py --check
  python -m unittest tests/test_enterprise_datacenter_dcim_stranded_power_srlg_fiber_capacity_milp_selection.py
  ```

### 5.2 核心量化对比图（碎片化搁浅电力占比 % vs. 主备光纤 SRLG 共管井违例率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Nameplate Excel + First-Fit", "metric": "Stranded Power Capacity (%)", "pct": 31.5 },
      { "tier": "Tier 1: Nameplate Excel + First-Fit", "metric": "SRLG Shared-Conduit Violation (%)", "pct": 18.4 },
      { "tier": "Tier 2: Prompt Rack/Patch LLM", "metric": "Stranded Power Capacity (%)", "pct": 28.0 },
      { "tier": "Tier 2: Prompt Rack/Patch LLM", "metric": "SRLG Shared-Conduit Violation (%)", "pct": 24.5 },
      { "tier": "Tier 3: Standard DCIM + SNMP", "metric": "Stranded Power Capacity (%)", "pct": 14.2 },
      { "tier": "Tier 3: Standard DCIM + SNMP", "metric": "SRLG Shared-Conduit Violation (%)", "pct": 6.8 },
      { "tier": "Tier 4: 5D Tetris + Suurballe SRLG", "metric": "Stranded Power Capacity (%)", "pct": 2.1 },
      { "tier": "Tier 4: 5D Tetris + Suurballe SRLG", "metric": "SRLG Shared-Conduit Violation (%)", "pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 cuts stranded data-center power to 2.1% and eliminates SRLG fiber violations (0.0%)",
    "subtitle": "Comparison across a 40 MW enterprise data center with 1,600 racks (lower is better)",
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

- [ ] **用“历史 `P99` 实测遥测功率包络 + 五维容量俄罗斯方块 `MILP`（电功率、液冷 `CDU`、42U 空间、地板承重、`3m DAC` 铜缆半径）”替代 100% 电源铭牌 Excel 扣减法**：最大化回收碎片化搁浅电力。
- [ ] **所有施工上架单（Cut-Sheet）必须精确指定到三相 `PDU` 的具体插座孔号（`Phase AB / BC / CA`）**：确保三相不平衡度 `<=2.5%`，且在单路供电故障（`A+B` 合并）时每一相电流严格 `<80% NEC` 持续负载上限。
- [ ] **主备双路（A/B）光纤跳线规划必须运行 `Suurballe` 共享风险链路组（`SRLG`）不相交算法**：严禁主备光纤共用任何一个弱电竖井、穿墙套管或配线架熔接盒。
- [ ] **在三维桥架曼哈顿距离公式中强制计入“机柜内垂直高度差 + 前后理线弯折余量（`0.6m`）”**：防止买来的 `3.0m` 无源铜缆（DAC）在现场差 20 厘米插不上。
- [ ] **设备通电前强制执行“驻场技师机柜正背面全景照片 `VLM` 竣工比对”**：自动核验网线 QR 标签与 `PDU` A/B 双路插孔无误后方可闭合供电开关。
