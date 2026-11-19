# 企业级石油化工乙烯裂解装置（Olefin Steam Cracker）、在线近红外原料族组成（`NIR PIONA`）前馈、辐射盘管结焦与管壁温度（`TMT <= 1085°C`）`MPC`、`IEC 61511 SIS` 联锁与大修（`TAR SIMOPS`）排程 Agent 架构选型指南：固定炉出口温度 PID+45天定周期烧焦 vs. 纯 LLM 读 DCS 报警日志 vs. 无约束黑盒神经网络 vs. 在线 `NIR PIONA` 前馈 + 自由基结焦动力学 `TMT MPC` + `SIL-3` 硬件钳位 + 大修动火空间隔离 `MILP` Agent

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-petrochemical-steam-cracker-coking-mpc-hazop-tar-agent-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、大型炼化一体化与百万吨级乙烯裂解厂总工程师、先进过程控制（`APC：AspenTech DMC3 / Honeywell Profit Controller / SPYRO`）负责人、过程安全（`IEC 61511 HAZOP / SIS`）及全厂大修（`Turnaround TAR`）指挥部的生产落地指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection/results.json`](../../experiments/enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection/results.json)。

![企业级石油化工乙烯裂解装置结焦动力学 MPC、PIONA 前馈与大修 SIMOPS Agent 架构总结图](../../assets/svg/enterprise-petrochemical-steam-cracker-coking-mpc-hazop-tar-agent-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在运营一座 **年产 `150 万吨` 乙烯的龙头石化装置（配置 `10` 台大型石脑油/乙烷蒸汽裂解炉、在线监控 `480` 根高温合金辐射炉管，且每次全厂停工大修 `Turnaround TAR` 涉及 `3,500` 张高危交叉作业工单）** 时，生产、安全与设备部门面临三大千万美元级痛点：
1. **石脑油原料罐切换时不管族组成（`PIONA`：正构烷烃/异构烷烃/环烷烃/芳烃）变化，死守固定炉出口温度（`COT`，Tier 1），导致高环烷烃重质油进炉瞬间剧烈结焦，单炉运行周期卡死在 `45 天` 被迫停炉烧焦（每年多停炉 `18.4 次`，双烯收率损失 `6.8%`）**！
2. **用无物理约束的黑盒神经网络（Tier 3）盲目提高裂解温度追逐短期乙烯收率，导致辐射盘管外壁温度（`TMT`）突破 `HP-40Nb` 耐热合金的 `1,085°C` 渗碳蠕变红线（引发 `$120 万` 爆管事故）；或者用纯大模型看 DCS 报警（Tier 2），在碳二加氢反应器飞温时错误建议“加大氢气流量”，直接诱发乙烯剧烈放热加氢飞温火炬放空**！
3. **全厂停工大修（`TAR`）3,500 张工单靠人工排计划（Tier 1），把“2# 裂解炉明火电焊（Hot Work）”和“隔壁急冷塔拆盲板放残油（Line Breaking）”排在同一时间同一区域，引发安全叫停窝工（大修关键路径延误高达 `21.5%`，每多停产一天少赚 `$280 万`）**！

在 **150 万吨/年乙烯装置、10 台裂解炉、480 根辐射炉管、3,500 张大修工单** 的石化基准规模下，四档石化裂解优化与大修安全架构的量化对决如下：

| 方案档位 | 核心架构组合 | 年均非计划停炉烧焦与跳车次数 | 双烯（乙烯+丙烯）收率损失率 | 辐射炉管渗碳爆管与飞温风险 | 全厂大修 (`TAR`) 关键路径延误率 | 月度收率流失、频繁烧焦与大修窝工总损失 (`TCO`) | 生产推荐度 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：固定炉出口温度 (`COT`) PID + 45天定周期烧焦** | 恒定 COT 设定值 + 满 45 天硬停炉烧焦 + 手工排 Primavera 大修表 | `18.4 次/年` ✕ | `6.8%` ✕ *(原料切罐过裂解)* | `3.4%` ✕ *(局部热斑盲区)* | `21.5%` ✕ *(动火与拆盲板冲突)* | `$1,350,000 / 月` ✕ | ⛔ 频繁停炉烧焦，大修交叉作业频频窝工 |
| **🟨 Tier 2：纯大模型直读 DCS 报警与交接班日志 (`Prompt LLM`)** | 把告警流塞给通用 LLM 推荐炉温与加氢反应器调参指令 | `16.2 次/年` ✕ | `7.5%` ✕ | `4.8%` ✕ *(加氢反应器飞温误导)* | `19.0%` ✕ | `$1,480,000 / 月` ✕ | ⛔ 违背放热反应动力学，有恶性飞温风险 |
| **🟦 Tier 3：无物理约束黑盒神经网络软测量 (`Black-Box ML`)** | 纯数据驱动追逐短期双烯收率（无结焦层导热方程与 `TMT` 硬钳位） | `6.4 次/年` | `2.9%` | `1.1%` ✕ *(超 1085°C 渗碳爆管)* | `9.2%` ✕ | `$490,000 / 月` ✕ | ⚠️ 为贪图 0.5% 短期收率把百万元炉管烧穿 |
| **🟩 Tier 4：在线 `NIR PIONA` 前馈 + 结焦动力学 `TMT MPC` + `SIL-3 SIS` + 大修 `SIMOPS MILP`** | **在线近红外 `PIONA` 前馈 + 单程结焦厚度/管壁温度 (`TMT<=1085°C`) `MPC` + `IEC 61511 SIL-3` 硬件联锁 + 大修 `3D SIMOPS MILP`** | **`0.4 次/年` 🏆** *(周期延至78天)* | **`0.5%` 🏆** *(-6.3 pts)* | **`0.0%` 🏆** *(零爆管零飞温)* | **`1.4%` 🏆** *(-20.1 pts)* | **`$110,000 / 月` 🏆** *(-91.9%)* | 🌟 **现代百万吨级大型石化装置唯一标准** |

> 💡 **架构师一句话选型铁律**：**“石化裂解装置决不能用固定炉温应对千变万化的石脑油罐，更绝不允许黑盒 AI 越过冶金极限定设定值——必须在进料总管部署在线近红外光谱（`Online NIR`）实时解析石脑油 `PIONA` 族组成前馈至 `SPYRO`/机理混合 `MPC`；按每组炉管的自由基结焦厚度微分方程动态平衡稀释蒸汽比与进料偏流，将管壁温度死死钳位在 `TMT <= 1,080°C`（运行周期从 45 天拉长至 78 天）；底层由物理隔离的 `IEC 61511 SIL-3 SIS` 安全仪表系统兜底防飞温；并在大修期间用 `3D SIMOPS MILP` 物理隔开 15 米内的动火电焊与带料拆盲板作业！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大切罐爆焦、加氢飞温与大修动火冲突死穴）

### Q1：为什么罐区切换一罐新采购的石脑油（Naphtha），裂解炉就会在几小时内迅速结焦、急冷锅炉（`TLE`）压差暴涨？“在线 `NIR PIONA` 前馈 `MPC`”怎么稳住？
- **大白话与第一性原理**：
  1. **同样叫“石脑油”，里面的分子裂解脾气天差地别**：上一罐是富含链烷烃（Paraffins `72%`）的优质轻油，下一罐切成环烷烃（Naphthenes `34%`）和芳烃偏高的重质油。如果还按 Tier 1 死守原来的固定炉出口温度（`COT = 845°C`）和固定稀释蒸汽比，环烷烃在高温下会迅速脱氢缩合生成多环芳烃（结焦前驱体），在炉管内壁疯狂长焦！
  2. **Tier 4 的“在线近红外（`NIR`）`PIONA` 族组成秒级前馈 + 裂解深度（`P/E Ratio`）恒定控制”**：
     - 在进料总管安装在线近红外光谱仪，每 60 秒解算出正构烷烃（P）、异构烷烃（I）、烯烃（O）、环烷烃（N）、芳烃（A）质量分数；
     - 在高环烷烃油品进炉前 15 分钟，`MPC` 自动前馈上调**水油比（Steam-to-Hydrocarbon Ratio `+0.08 kg/kg`）**降低烃分压，并精准微调 `COT`，将双烯收率损失从 `6.8%` 压至 **`0.5%`**！

### Q2：辐射炉管是怎么被烧穿的？为什么黑盒机器学习（Tier 3）会惹出 `$120 万` 的爆管事故，而 Tier 4 能把清焦周期从 45 天延长到 78 天且 `0.0%` 爆管？
- **焦层热阻传热定律与 `HP-40Nb` 高温合金 `1,085°C` 渗碳蠕变铁闸**：
  - 裂解炉管内部一旦沉积厚度为 $\delta_{\text{coke}}$ 的焦炭层（导热系数极低 $\lambda_{\text{coke}} \ll \lambda_{\text{metal}}$），为了维持管内油气达到 `840°C` 裂解温度，炉膛烧嘴必须猛烈加热管外壁，导致管壁金属温度（`TMT`）按如下物理公式急剧攀升：
    $$T_{\text{TMT}, k} = T_{\text{gas}} + q_{\text{flux}} \cdot \left(\frac{\delta_{\text{coke}, k}}{\lambda_{\text{coke}}} + \frac{d_{\text{wall}}}{\lambda_{\text{metal}}}\right) \le 1,085^\circ\text{C}$$
  - Tier 3 黑盒模型只看到“升温能涨乙烯收率”，全然不知第 3 组炉管焦层已厚达 `4.8mm`、外壁 `TMT` 已逼近 `1,085°C` 渗碳极限！
  - Tier 4 融合**红外测温枪 + 文丘里喉管音速压降（`Venturi ΔP`，防窥视孔积碳遮挡误读）+ 急冷锅炉（`TLE`）入口温差**，当单组炉管 `TMT` 达 `1,072°C` 时，单程偏流 `MPC` 自动将该组进料微降 `-4%` 并补入稀释蒸汽抑制自由基结焦速率，使全炉均匀长焦，**将运行周期从 45 天推至 78 天（年停炉从 `18.4 次` 降至 `0.4 次`）**！

### Q3：为什么碳二加氢除炔反应器出现 `+6°C` 热点时，纯大模型（Tier 2）给出的建议会酿成火炬恶性放空？大修 `3,500` 张工单怎么防空间碰撞？
- **`IEC 61511 SIL-3` 独立安全仪表硬件钳位 + 大修 `3D SIMOPS`（交叉作业）时空 `MILP`**：
  1. **加氢除炔飞温硬联锁**：碳二加氢反应器旨在把微量乙炔（`C2H2`）选择性加氢为乙烯；一旦床层出现 `+6°C` 放热热点且 CO 抑制剂跌至 `0.3 ppm`，若听信通用大模型“加大氢气吹扫”，海量乙烯会被非选择性加氢为乙烷，释放出 3 倍剧烈反应热瞬间引发飞温（Thermal Runaway）！Tier 4 通过机理 `MPC` 立即下调氢炔摩尔比，且底层由**物理隔离的 `IEC 61511 SIL-3 SIS` 硬件 PLC** 锁死温升速率上限！
  2. **大修 `3D SIMOPS` 动火与拆盲板 15 米安全隔离**：在 28 天大修 `MILP` 中，结合装置三维点云坐标强制执行 `API RP 2219` 铁律——**凡某管线正在执行打开法兰/抽堵盲板（Line Breaking），其三维半径 `15.0 米` 及正下方垂直投影区内物理禁止签发任何明火电焊（Hot Work）作业票**，同时协同 600 吨履带吊车走行路径，将大修延误率从 `21.5%` 砍至 **`1.4%`**！

---

## ③ 🧭 直观可视化图解（决策树 + 裂解/SIS/大修架构图 + 切罐前馈与防爆管时序图）

### 3.1 企业级石化乙烯裂解控制、过程安全与大修排程架构选型决策树

```mermaid
flowchart TD
    A["企业级百万吨乙烯裂解装置控制、SIS 安全与大修排程<br/>(150 万吨/年乙烯 + 10 台裂解炉 + 480 根炉管 + 3,500 张大修工单)"] --> B{"是否仍用固定 COT 炉温 PID + 45 天硬烧焦<br/>或让纯大模型读 DCS 报警给调参建议?"}
    B -- "是 (固定炉温或裸 LLM 看日志)" --> C["⛔ 频繁结焦停炉、加氢飞温与大修窝工 (Tier 1 / Tier 2)<br/>年均 18.4 次非计划烧焦，6.8% 双烯收率流失，大修延误高达 21.5%"]
    B -- "引入先进控制与软测量" --> D{"是否融合在线 NIR PIONA 前馈、结焦动力学 TMT (<=1085°C) MPC、<br/>IEC 61511 SIL-3 硬件钳位与大修 3D SIMOPS MILP?"}
    D -- "否 (无约束黑盒神经网络)" --> E["⚠️ 炉管超温渗碳爆管风险 (Tier 3)<br/>盲目追收率推高管壁温度逼近 1085°C，仍有 1.1% 爆管风险与 9.2% 大修延误"]
    D -- "PIONA 前馈 + 结焦 TMT MPC + SIL-3 + SIMOPS MILP" --> F["🟩 Tier 4: PIONA 前馈 + 结焦 TMT MPC + SIL-3 SIS + 大修 MILP 🏆<br/>(78 天长周期 | 0.5% 收率损失 | 0.0% 爆管风险 | 1.4% 大修延误)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 在线 `NIR PIONA` 前馈 + 结焦动力学 `TMT MPC` + `SIL-3 SIS` + 大修 `MILP` 架构图

```mermaid
flowchart LR
    subgraph Feed["1. 在线近红外 PIONA 前馈层"]
        NIR["🔬 进料总管在线 NIR 光谱仪<br/>60s 实测烷/环烷/芳烃 (PIONA)"] --> MPC["⚙️ SPYRO 机理 + 结焦动力学混合 MPC<br/>动态调节水油比与单程偏流<br/>(双烯收率损失仅 0.5%)"]
    end

    subgraph Furnace["2. 炉管 TMT (<=1085°C) 与 SIL-3 硬件门禁"]
        MPC --> Coil["🔥 红外 TMT + 文丘里压降 ΔP + TLE 温差<br/>锁死 HP-40Nb 管壁 <=1,080°C<br/>(运行周期从 45 天拉长至 78 天)"]
        Coil --> SIS["🛡️ IEC 61511 SIL-3 独立硬件 PLC<br/>加氢除炔防飞温物理联锁 (0.0% 事故)"]
    end

    subgraph TAR["3. 全厂大修 3D SIMOPS 与吊车 MILP"]
        WO["🗓️ 3,500 张大修工单 + 盲板清单"] --> SIMOPS["🚧 3D 空间交叉作业 (SIMOPS) MILP<br/>禁拆盲板 15m 内动火 + 600t 吊车排程<br/>(关键路径延误仅 1.4%)"]
    end
```

### 3.3 从“高环烷烃石脑油切罐前馈”到“单炉管 1,072°C 抑焦延寿与大修防冲突”时序图

```mermaid
sequenceDiagram
    autonumber
    participant Tank as 原料罐区 & 在线 NIR 光谱仪
    participant MPC as 裂解深度与结焦动力学 MPC
    participant SIS as 4# 裂解炉 / 碳二加氢 SIL-3 SIS
    participant TAR as 全厂大修 3D SIMOPS 排程 MILP

    Tank->>MPC: 切换至高环烷烃石脑油罐：在线 NIR 检出环烷烃从 18% 跃升至 34%（预计 15 分钟后进炉）
    MPC->>SIS: ⚡ 前馈上调稀释蒸汽比 +0.08 kg/kg 并微调 COT，稳住丙烯/乙烯比 P/E = 0.54，抑制芳烃缩合结焦！
    SIS-->>MPC: 运行第 52 天：4# 炉第 3 程红外测温 + 文丘里 ΔP 联合确认焦层厚 4.8mm、外壁 TMT = 1,072°C (逼近 1,085°C 极限)
    MPC->>SIS: 单程偏流控制：下调第 3 程烃进料 -4% 并补入蒸汽 -> 将 TMT 压回 1,064°C，成功续命跑满 78 天长周期！
    TAR->>TAR: 校验大修工单 #WO-1142（2# 炉电焊动火）：检出距离隔壁急冷塔带料拆盲板作业仅 8.2m (<15m 红线)
    TAR-->>SIS: 🚨 自动锁定并推迟动火电子作业票至盲板紧固气密合格后，消除爆炸隐患且保障 600t 主吊车零等待！
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：固定 COT PID + 45天烧焦 | 🟨 Tier 2：纯 LLM 读 DCS 日志 | 🟦 Tier 3：无约束黑盒神经网络 | 🟩 Tier 4：`PIONA` 前馈 + 结焦 `MPC` + `SIL-3` |
|---|---|---|---|---|
| **年均非计划停炉烧焦与跳车次数** | `18.4 次/年` ✕（45天硬停炉） | `16.2 次/年` ✕ | `6.4 次/年` | 🏆 **`0.4 次/年`（运行周期延长至 `78 天`，降幅 `97.8%`）** |
| **双烯（乙烯+丙烯）收率损失率** | `6.8%` ✕（切罐过裂解或欠裂解） | `7.5%` ✕ | `2.9%` | 🏆 **`0.5%`（在线 `NIR PIONA` 秒级前馈 + `SPYRO` 寻优）** |
| **辐射炉管渗碳爆管与飞温风险** | `3.4%` ✕（人工巡检滞后） | `4.8%` ✕（加氢飞温反向误导） | `1.1%` ✕（贪收率超 `1085°C`） | 🏆 **`0.0%`（单程 `TMT <= 1080°C` 软约束 + `SIL-3` 硬钳位）** |
| **全厂大修 (`TAR`) 关键路径延误率** | `21.5%` ✕（现场动火冲突叫停） | `19.0%` ✕ | `9.2%` ✕ | 🏆 **`1.4%`（`3D SIMOPS` 15m 动火隔离 + 吊车 `MILP`）** |
| **月度收率、烧焦与大修总损失 (`TCO`)** | `$1,350,000 / 月` | `$1,480,000 / 月` | `$490,000 / 月` | **`$110,000 / 月` 🏆（节省 91.9%）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（150 万吨/年乙烯裂解装置 / 10 台裂解炉 / 480 根炉管 / 3,500 张大修工单）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级石油化工乙烯裂解结焦 MPC 与大修排程定容计算器](../reports/enterprise-petrochemical-steam-cracker-coking-mpc-hazop-tar-agent-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-petrochemical-steam-cracker-coking-mpc-hazop-tar-agent-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection/run.py --check
  python -m unittest tests/test_enterprise_petrochemical_steam_cracker_coking_mpc_hazop_tar_agent_selection.py
  ```

### 5.2 核心量化对比图（双烯收率损失率 % vs. 全厂大修关键路径延误率 TAR Slip %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Fixed COT PID + 45d Run", "metric": "Olefin Yield Loss (%)", "pct": 6.8 },
      { "tier": "Tier 1: Fixed COT PID + 45d Run", "metric": "Turnaround TAR Slip (%)", "pct": 21.5 },
      { "tier": "Tier 2: Prompt DCS Log LLM", "metric": "Olefin Yield Loss (%)", "pct": 7.5 },
      { "tier": "Tier 2: Prompt DCS Log LLM", "metric": "Turnaround TAR Slip (%)", "pct": 19.0 },
      { "tier": "Tier 3: Unconstrained Yield ML", "metric": "Olefin Yield Loss (%)", "pct": 2.9 },
      { "tier": "Tier 3: Unconstrained Yield ML", "metric": "Turnaround TAR Slip (%)", "pct": 9.2 },
      { "tier": "Tier 4: PIONA + TMT MPC + SIMOPS", "metric": "Olefin Yield Loss (%)", "pct": 0.5 },
      { "tier": "Tier 4: PIONA + TMT MPC + SIMOPS", "metric": "Turnaround TAR Slip (%)", "pct": 1.4 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 cuts olefin yield loss to 0.5% and turnaround schedule slip to 1.4%",
    "subtitle": "Comparison across a 1.5M tonnes/yr ethylene steam-cracker complex (lower is better)",
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

- [ ] **石脑油/轻烃进料总管必须配置在线近红外光谱（`Online NIR`）实时解析 `PIONA` 族组成并前馈至裂解炉 `MPC`**：在切罐油品性质突变前提前调整稀释蒸汽比与 `COT`，避免过裂解结焦。
- [ ] **辐射盘管外壁金属温度（`TMT`）必须采用“光学红外测温 + 文丘里喉管绝压降（`Venturi ΔP`）+ 急冷锅炉（`TLE`）入口温差”三冗余软测量**：防止炉膛看火孔积碳遮挡导致光学测温偏低而烧穿 `HP-40Nb` 炉管（硬红线 `<=1,085°C`）。
- [ ] **所有 AI 与先进控制（`APC / MPC`）设定值下方必须保留物理隔离的 `IEC 61511 SIL-3 SIS` 安全仪表系统硬件钳位**：严禁上层算法直接跨过 `SIS` 操纵碳二加氢反应器氢气阀门或裂解炉燃料气总阀。
- [ ] **全厂停工大修（`Turnaround TAR`）排程与电子作业票（`e-PTW`）必须内置三维空间 `SIMOPS`（交叉作业）硬门禁**：严格禁止在拆盲板/动设备吹扫点 `15 米` 半径及垂直下方同时签发明火电焊作业票。
- [ ] **烧焦（Decoking）过程必须根据尾气在线 `CO / CO2` 红外分析仪实行自适应空气-蒸汽配比控制**：防止炉管内残焦燃烧过猛引起局部超温渗碳。
