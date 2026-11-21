# 大型 AI/HPC 加速器集群托管机房（`Partner / Colocation Data Center`）选址资格认证（`Site Qualification`）、上电开通（`Turn-Up`）、`45–120kW+` 冷板液冷验收与运维就绪全生命周期架构指南

> **适用读者**：负责将 Google Cloud TPU（`v5p / TPU7x Ironwood / TPU 8t & 8i`）或 NVIDIA 高密 GPU（`GB200 NVL72 / B200 / H200`）落地到合作伙伴托管机房（Partner-Operated Colocation Sites）、主权云机房或企业自建智算中心的 Cloud & AI Infrastructure Solution Architect、HPC 交付总工与数据中心基础设施（DCIM）负责人。
> **核心课题**：当最新一代 AI 加速器从云厂商自营机房走向合作伙伴托管机房与企业本地机房（单机柜功率飙升至 **`45–120kW+`**、单柜满载重量突破 **`1,800–2,200 kg`**、包含数万芯 MPO-16 高速光纤与冷板液冷二次侧环路）时，为什么沿用传统云计算“插电 Ping 通能 SSH 就移交客户（Tier 1）”或“仅做单机 `nvidia-smi / DCGM` 冒烟测试（Tier 2）”会导致 **从设备到货至稳定投产耗时 `68.0 天`、首月高达 `34.5%` 的作业因早期硬件夭折（Infant Mortality）与光纤端面污染中断、跨机集合通信带宽仅达标称值的 `61.0%`（每月造成 `$920,000` 的算力空置、返工拔线与停机损失）**？如何通过 **“8 阶段刚性交付门禁（`Site Qual -> Rack/DLC -> Rigging -> Hookup -> LLDP/Optical Scope -> Firmware/Redfish -> 48h Burn-In -> Handover`）+ `CDU` 液冷/供电与计算 Pod 故障域严格对齐 + `>=90%` 理论带宽（`NCCL busbw / TPU ICI`）验收门槛 + 四层遥测联动与 `3%` 现场热备件池”** 四级架构，将上电开通周期压缩至 **`9.5 天`**、首月作业中断率压降至 **`0.6%`**、集群集合通信带宽达标率提升至 **`94.8%`**，并将月度综合异常与运维 TCO 降低 **`89.7%`（`$920,000 -> $95,000/月`）**？

![大型 AI/HPC 加速器集群托管机房选址资格认证、上电开通与验收就绪全生命周期架构总结图](../../assets/svg/ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

在百兆瓦级高密 AI 智算中心交付中，**“机器搬进机房亮灯能 SSH”只完成了整个交付工作量的 20%——真正的生死线在于：卸货电梯与地板会不会被 2.2 吨的液冷机柜压塌、120kW 冷板里的一个微小气囊会不会让芯片秒级飙到 95°C 降频、上万根 MPO 光纤里混入的 3 颗灰尘和 2 根插反的跨机柜线缆会不会让千卡 AllReduce 慢 5 倍**：

| 交付与机房选型层级 | 适用场景 | 到货至稳定投产周期 (`Turn-Up Days`) | 首月早期夭折作业中断率 (`Month-1 Aborts`) | 跨机集合通信实测带宽占比 (`busbw / ICI`) | 月度空置延误、返工拔线与停机总成本 (`TCO`) | 最终拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：传统风冷托管机房 (`<20kW/柜`) + Ping/SSH 通即移交（反模式）** | 仅适合 `<12kW` 传统 CPU Web/数据库服务器 | **`68.0 天`（反复返工排障！）** | **`34.5%`（新卡 DOA 与光模块闪断）** | `61.0%`（光纤污染+热降频） | `$920,000 / 月` | ❌ **灾难级烂尾**：一跑满载就跳闸/过热，AI 团队成了全职网管 |
| **Tier 2：后置背门热交换改造 (`RDHx <=40kW`) + 单节点冒烟测试** | 中小规模 `L40S / PCIe H100` 过渡集群 | `42.0 天` | `16.8%` | `74.5%`（漏检跨柜插错线！） | `$560,000 / 月` | ❌ **单机合格、集群瘫痪**：单机测试完全测不出跨柜 Rail/Torus 线序接反 |
| **Tier 3：`120kW` 冷板液冷 (`DLC`) 但未对齐 `CDU` 故障域且无 48h 烤机** | 硬件到位但缺乏标准化验收门禁的新建机房 | `24.0 天` | `7.8%`（漏放早期 HBM 坏块） | `84.0%` | `$310,000 / 月` | ⚠️ **故障爆炸半径失控**：1 台 CDU 检修导致 4 个独立训练 Slice 同时陪葬 |
| **Tier 4：8 阶段交付门禁 + `120kW DLC/CDU` 故障域对齐 + `48h` 满载烤机 (`>=90% busbw/ICI`) + 四层遥测（推荐）** | **`TPU7x Ironwood / v5p / 8t & 8i` 与 `GB200 NVL72 / B200` 托管/本地机房主力标准** | **`9.5 天`（提速 `7.2 倍`）** | **`0.6%`（98% 坏件在移交前剔除）** | **`94.8%`** | **`$95,000 / 月`（省 `89.7%`）** | 🏆 **生产金标准**：移交当天即满血跑满千卡训练，零隐性暗病 |

### 💡 架构师的三大黄金铁律
1. **严格执行从选址到移交的“8 阶段刚性门禁（8-Stage Turn-Up Lifecycle）”，前一阶段未签字绝不进入下一阶段**：
   * **阶段 1（选址资格认证 `Site Qualification`）** $\rightarrow$ **阶段 2（机柜落位与供电/液冷包络设计 `Rack & DLC Envelope`）** $\rightarrow$ **阶段 3（到货接收、卸货平台与承重动线 `Receiving & Rigging`）** $\rightarrow$ **阶段 4（机柜就位、母线槽与液冷歧管连接 `Racking & Manifold Hookup`）** $\rightarrow$ **阶段 5（结构化布线、光纤端面镜检与 `LLDP` 逐跳拓扑审计 `Cabling & LLDP Audit`）** $\rightarrow$ **阶段 6（固件基线锁死与带外 `BMC Redfish` 纳管 `Firmware & OOB`）** $\rightarrow$ **阶段 7（`48 小时` 100% TDP 热稳态与集合通信验收 `Burn-In & Acceptance`）** $\rightarrow$ **阶段 8（四层监控、现场备件池与运维移交 `Operational Handover`）**。
2. **机房物理故障域（供电母线 `Busway` + 液冷分配单元 `CDU`）必须与计算逻辑拓扑（`TPU 4x4x4 Cube` / `GB200 NVL72 Pod` / `SU`）1:1 严格对齐**：
   * 绝不能按传统机房习惯把一排 `CDU` 液冷支管横跨连接 4 个不同训练 Slice 的机柜！必须保证 **“1 组 `N+1` 冗余 `CDU` 环路 + 1 对 `A/B` 双路供电母线”严格对应整数个独立的计算故障域（Fault Domain）**，且配备硬件级 **绳式/光电漏液传感器联锁分励脱扣（Leak Detection Shunt-Trip Interlock）**。
3. **没有通过“`48h` 方波热冲击 + `>=90%` 理论线速集合通信（`NCCL busbw` / `TPU ICI`）+ 参考大模型步时方差 `<=2.5%`”的集群，绝不允许签字移交算法团队**：
   * 约 **`85%` 的光模块早期劣化、冷板导热硅脂（TIM）贴合不良与 HBM 边缘坏块** 只会在持续满载升温后的前 `24–48 小时` 内暴露。必须在 Stage 7 一次性逼出全部早期夭折硬件（Infant Mortality）并从现场 `3%` 预烤热备件池中原地替换！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A

### Q1：去考察一家合作托管机房（Partner / Colocation DC）是否能放 `TPU7x Ironwood` 或 `GB200 NVL72`，架构师现场必须核查哪 6 项硬指标？
* **高密 AI 机房选址资格认证（Site Qualification）六大生死线**：
  1. **单柜功率密度与变压器裕量（`45–120kW+ / Rack`）**：机房不仅要有足够的總兆瓦数（MW），还必须支持机柜上方双路 `415V/480V Three-Phase Busway`（智能母线槽）及应对训练步同步启停带来的 **`dP/dt` 瞬态功率阶跃**。
  2. **冷板液冷（Direct-to-Chip DLC）与 `CDU` 二次侧水质/流量**：每 `100 kW` 热负荷在 `10°C` 温差下需要约 **`150 LPM`（升/分钟）** 的 `PG25`（25% 丙二醇去离子水）流量；核查一次侧冷冻水（`FWS`）供水温度（支持 `25°C–32°C` 温水冷却可大幅降低 PUE）、`N+1 CDU` 冗余水泵、**`25 微米` 高精度过滤器**、自动排气阀（Air Purge）及机柜底部漏液托盘与告警联锁。
  3. **结构承重与运输动线（`Floor Loading >= 2,000–2,500 kg/m²` & `Rigging Path`）**：满载液冷机柜连带冷却液与铜背板重达 **`1.8–2.2 吨`**！必须核验从卸货月台升降机、货梯轿厢尺寸/载重（`>=3,000 kg`）、走廊转弯半径、门口高度（`>=2.4m` 无门槛）到机房地板均布与点荷载（必要时加装散力钢梁 `Spreader Plates`）。
  4. **机柜间距与全光/铜缆物理走线距离限制**：`3D Torus` 跨柜高速有源铜缆（AEC）有严格的米数上限（通常 `<3–5m`），机柜必须采用紧凑成组（Pod/Row）布局，上走线光纤桥架需提供双层弯曲半径保护（`>=30mm`）。
  5. **双物理路由外联光纤与专线（`Dual-Diverse WAN / Cloud Interconnect`）**：检查东西两侧独立进局管道（Diverse Entry Conduits），提供到云端对象存储或核心网的低延迟 `400G/800G` 波分/专线，以支撑 `Multislice DCN` 与训练数据流。
  6. **驻场工程服务（`24x7 Remote Hands SLA`）与安全保卫**：是否有受过 MPO 光纤清洁与快换街头（Blind-Mate QD）培训的驻场工程师，P0 硬件故障能否做到 **`<15 分钟` 到场响应**。

### Q2：为什么说“人工插完几千根光纤后不做自动化 `LLDP` 与光功率计审计，集群通信性能一定会腰斩”？
* **两大致命隐患：跨机柜线序插错（Cabling Mis-wire）与光纤端面微尘污染（Dirty Ferrule）**：
  * **线序插错**：在一个 64 机柜集群中需要端接超过 `8,000` 根光纤。只要施工队把第 12 号柜 `NIC-3` 去往 Leaf-3 的光纤和去往 Leaf-4 的光纤插反，单机 Ping 测试依然 100% 正常，但一跑跨机 `Rail-Optimized AllReduce` 或 `3D Torus Ring` 就会发生跨平面绕行与严重拥塞，导致千卡带宽跌至 `74.5%`！**解法**：Stage 5 强制运行自动化脚本采集全网交换机/网卡的 **`LLDP` 邻居表**，与设计接线表（`Cutsheet YAML`）做 `diff`，不一致项必须归零！
  * **光纤端面灰尘**：`800G/400G DR8` MPO-16 光模块对灰尘极度敏感，一粒肉眼看不见的皮屑就会让光衰增加 `2.5 dB`、触发高频 `Pre-FEC BER` 重传甚至烧毁激光器。**解法**：施工遵循 **“每次插拔必用端面显微镜检 + 一键清洁笔（Inspect -> Clean -> Inspect）”**，并在 Stage 5 自动化读取所有光模块 `DOM` 接收光功率（`Rx Power`）与 `OCS` 插入损耗（`<1.8 dB`）。

### Q3：GPU 集群与 TPU 集群在 Stage 7（压测验收 Burn-In & Acceptance）的具体定量及格线分别是什么？
* **四级递进压测漏斗（从单芯片 → 单节点 → 跨柜集合通信 → 端到端模型步时）**：

| 验收层级 | NVIDIA GPU 集群定量通过门禁 (`H200 / B200 / GB200`) | Google Cloud TPU 集群定量通过门禁 (`v5p / v6e / TPU7x / 8t & 8i`) |
| :--- | :--- | :--- |
| **L1：单芯片与显存健康** | `dcgmi diag -r 4` 零报错；`0` 个 Uncorrectable ECC、`0` 个新增 Row-Remap、`0` 个致命 XID (`48/63/79/94`) | 设备自检 `libtpu` 零硬件异常；`TensorCore / SparseCore` 与 `192GB HBM3e` ECC 校验 `0` 不可纠错误 |
| **L2：48h 满功率热冲击** | `HPL` 或 `DGEMM/FP8` 方波满载连续烤机 `48 小时`：所有 GPU 核心温度 `<=83°C`、HBM `<=88°C`、`0%` 热降频（Thermal Throttle） | 满载矩阵乘方波烤机 `48 小时`：冷板出水温度波动 `<±1.5°C`、芯片结温无降频、冷却支路压差正常 |
| **L3：高速互连与集合通信** | 机内 `nvbandwidth` 达标；跨机 `nccl-tests` (`all_reduce_perf`, `alltoall_perf` @ `8GB`) **实测 `busbw >= 90%` 理论线速**，且两两节点二分测试零慢链路 | `3D Torus ICI` 全轴与 Wrap-Around 回环链路测试 + `OCS` 光路切换测试通过；**`All-Reduce / All-Gather` 达标率 `>=92%`**，Post-FEC `BER < 1e-12` |
| **L4：参考大模型端到端步时** | 运行标准 `Llama-3-70B / DeepSeek-671B` 1,000 步：MFU 达到基线 `>=98%`，不同子集群间步时标准差 `<=2.5%` | 运行官方 `MaxText` 参考训练作业与 `vLLM-TPU` 基准 1,000 步：步时方差 **`<=2.5%`**，且完成 `Orbax` Checkpoint 读写闭环 |

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 8 阶段交付架构 + 时序图）

### 3.1 大型 AI/HPC 托管机房上电开通（Turn-Up）8 阶段门禁流程图

```mermaid
flowchart TD
    S1["阶段 1: 选址资格认证 (Site Qualification)\n核验 45-120kW/柜电力、CDU 液冷流量、>=2000kg/m² 承重与双路光纤"] --> S2["阶段 2: 机柜落位与故障域对齐设计\n1 组 N+1 CDU + A/B 母线严格对齐 1 个 TPU Cube / NVL72 Pod"]
    S2 --> S3["阶段 3: 到货接收与重型搬运 (Rigging)\n防震倾角核验、散力钢梁铺设、2.2 吨整机柜安全落位"]
    S3 --> S4["阶段 4: 水电歧管接入与保压检漏\n25um 过滤冲洗、氮气/冷却液 24h 保压、漏液传感器联锁测试"]
    S4 --> S5["阶段 5: 结构化布线与自动化 LLDP/光衰审计\nMPO 端面镜检 + 脚本比对 Cutsheet YAML 100% 零错连 + BER < 1e-12"]
    S5 --> S6["阶段 6: 固件黄金基线与带外 Redfish 纳管\n统一刷写 BMC / BIOS / NIC / GPU / Switch 固件并接入四层监控"]
    S6 --> S7["阶段 7: 48 小时满载烤机与集合通信验收\nDCGM L4 / HPL 热冲击 + NCCL busbw / TPU ICI >= 90% + MaxText 步时验证"]
    S7 --> S8["阶段 8: 生产移交与运维就绪 (Handover)\n锁定 3% 现场预烤热备件池 + <15m Remote Hands SLA + 自动故障隔离 Runbook"]
```

### 3.2 Tier 4 物理设施、算力 Pod 故障域对齐与四层全栈遥测架构图

```mermaid
flowchart LR
    subgraph Facility["🏢 第 1 层: 机房设施与液冷层 (BMS / Modbus)"]
        FWS["一次侧冷冻水 (FWS)\n+ 双路 A/B 高压母线槽"] --> CDU["N+1 冗余液冷分配单元 (CDU)\nPG25 流量/压差/25um 过滤器 + 漏液脱扣联锁"]
    end

    subgraph Hardware["🖥️ 第 2 & 3 层: 硬件带外与高速互连层 (Redfish + Fabric)"]
        CDU ==>|"1:1 故障域对齐"| POD["算力 Pod (TPU 4x4x4 Cube / GB200 NVL72)\nBMC Redfish 温度/电压/功耗监控"]
        POD <==>|"MPO-16 光纤 / AEC"| FAB["3D Torus + OCS / IB-RoCE 交换网\n实时采集 DOM Rx 光功率 / Pre-FEC BER / PFC"]
    end

    subgraph Workload["🚀 第 4 层: 调度与工作负载层 (Prometheus + Auto-RMA)"]
        POD & FAB --> MON["四层关联可观测中台 (Prometheus / VictoriaMetrics)\n关联: 冷却水温 -> 芯片温度 -> 光口误码 -> 训练 Step Time"]
        MON --> RMA["自动 Cordon 隔离慢节点 + 触发 <15m 驻场备件更换 (3% 现场热备池)"]
    end
```

### 3.3 全链路 Stage 5–8 自动化布线核验、48h 烤机排障与移交时序图

```mermaid
sequenceDiagram
    autonumber
    participant Tech as 驻场交付工程师 (Remote Hands)
    participant Bot as Turn-Up 自动化验收引擎
    participant BMC as 带外 BMC Redfish & 交换机
    participant Cluster as 64 机柜 AI 集群 (TPU/GPU)

    Tech->>BMC: 完成机柜水电接入、液冷排气与 8,000 根光纤端接
    Bot->>BMC: [Stage 5] 拉取全网 `LLDP` 邻居表与光模块 `DOM Rx Power`
    BMC-->>Bot: 告警: `Rack-14 Node-3 Port-2` 与 `Port-3` 光纤反接；`Rack-22 Port-7` 光衰 `-6.8 dBm` (端面污染!)
    Bot->>Tech: 自动生成精准整改工单 (精确到机柜 U 位与端口号)
    Tech->>BMC: 清洁光纤端面并纠正反接线序 -> 重新触发 Stage 5 (`100% PASS`)
    Bot->>BMC: [Stage 6] 批量校对并锁死 `BMC / BIOS / NIC / Accelerator` 黄金固件版本
    Bot->>Cluster: [Stage 7] 启动 `48h` 方波热冲击 + 二分法 `NCCL / TPU ICI` 集合通信压测 + `MaxText/Llama` 基准
    Cluster-->>Bot: 第 19 小时检出 `Rack-08 Node-2 GPU-5` 出现新增 `HBM Uncorrectable ECC`
    Bot->>Tech: 自动隔离该节点并调用现场 `3%` 预烤热备件池在 35 分钟内完成更换重测
    Bot->>Cluster: 复测全集群 `busbw = 94.8% (>=90%)`、步时方差 `1.4% (<=2.5%)` -> 签发《Stage 8 生产移交证书》
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：传统风冷机房 + Ping/SSH 通即移交 | Tier 2：背门改造 (`RDHx`) + 单节点冒烟测试 | Tier 3：`120kW` 冷板液冷（无故障域对齐/无 48h 烤机） | Tier 4：8 阶段门禁 + `120kW DLC/CDU` 对齐 + `48h` 烤机 (`>=90% busbw`) + 四层遥测（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **到货至稳定投产周期 (`Turn-Up`)** | **`68.0 天`（边跑训练边停机修线换卡）** | `42.0 天` | `24.0 天` | **`9.5 天`（标准化流水线一次点亮交钥匙）** |
| **首月早期夭折作业中断率 (`M1 Aborts`)** | **`34.5%`（新卡 DOA 与光模块闪断频发）** | `16.8%` | `7.8%` | **`0.6%`（`98%` 早期缺陷在 48h 烤机期被拦截替换）** |
| **集合通信实测带宽达标率 (`busbw/ICI`)** | `61.0%` | `74.5%`（漏检跨柜接反与微尘） | `84.0%` | **`94.8%`（强制 `>=90%` 线速与 `LLDP` 零错连门禁）** |
| **`CDU` 液冷与电力故障爆炸半径** | 热岛效应随机宕机 | 局部过热降频 | ❌ 1 台 `CDU` 停机波及 4 个不同训练 Slice | **✅ `CDU + A/B Busway` 与计算 Pod 1:1 物理对齐** |
| **全栈遥测与硬件故障自愈 (`RMA`)** | 仅看主机 Ping/CPU | 仅看单机 `nvidia-smi` | 设施 BMS 与算力监控割裂 | **✅ 四层联动（`BMS + Redfish + Fabric + XProf/DCGM`）+ `3%` 现场热备** |
| **月度空置延误、返工与停机总 TCO** | `$920,000 / 月` | `$560,000 / 月` | `$310,000 / 月` | **`$95,000 / 月`（节省 `89.7%`）** |

---

## 5. 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 基准算例假设（`64` 机柜高密液冷智算集群 / 单柜 `120 kW` / 总 IT 负荷 `7.68 MW`）
* **业务规模**：在合作伙伴托管机房部署 `64` 台 `120 kW` 冷板液冷高密机柜（如 `TPU7x Ironwood` 64-Chip Cubes 或 `NVIDIA GB200 NVL72` 机柜，总 IT 功率 `7.68 MW`，二次侧冷却液总流量 `11,520 LPM`）。
* **工程实测与仿真代码**：见 [`experiments/ai_hpc_colocation_datacenter_site_qualification_turnup_readiness_selection/run.py`](../../experiments/ai_hpc_colocation_datacenter_site_qualification_turnup_readiness_selection/run.py) 与 [`results.json`](../../experiments/ai_hpc_colocation_datacenter_site_qualification_turnup_readiness_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式容量与 TCO 计算器**：📊 [打开大型 AI/HPC 托管机房选址、液冷流量与上电验收计算器 (`ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-sizer.html`)](../reports/ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-selection-cards.html`)](../reports/ai-hpc-colocation-datacenter-site-qualification-turnup-readiness-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Air-Cooled + Ping-Only Handover", "turnup_days": 68.0, "abort_pct": 34.5, "busbw_pct": 61.0, "monthly_tco_k": 920.0 },
      { "tier": "Tier 2: RDHx + Single-Node Smoke Test", "turnup_days": 42.0, "abort_pct": 16.8, "busbw_pct": 74.5, "monthly_tco_k": 560.0 },
      { "tier": "Tier 3: DLC Without CDU Alignment / 48h Soak", "turnup_days": 24.0, "abort_pct": 7.8, "busbw_pct": 84.0, "monthly_tco_k": 310.0 },
      { "tier": "Tier 4: 8-Stage Gate + CDU Align + 48h Soak", "turnup_days": 9.5, "abort_pct": 0.6, "busbw_pct": 94.8, "monthly_tco_k": 95.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "turnup_days": "Quantity",
    "abort_pct": "Percentage",
    "busbw_pct": "Percentage",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Compresses Turn-Up to 9.5 Days, Hits 94.8% Busbw, and Cuts Monthly TCO by 89.7%",
    "subtitle": "Monthly post-handover downtime, re-cabling rework, and cooling inefficiency TCO ($ USD Thousands)",
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

- [ ] **阶段 1–3 物理机房与重型动线核查**：卸货平台、货梯载重（`>=3,000 kg`）、通道净高（`>=2.4m`）与机房地板承重（`>=2,000–2,500 kg/m²` 或已铺设散力钢梁）是否已由结构工程师书面签字确认？
- [ ] **阶段 2 & 4 冷板液冷（`DLC`）与故障域 1:1 对齐**：二次侧 `PG25` 回路是否已完成 `25 微米` 冲洗过滤与 `24 小时` 保压测试，且确保单组 `N+1 CDU` 与双路 `A/B` 母线严格对应单一计算 Pod/Slice？
- [ ] **阶段 5 自动化 `LLDP` 接线表比对与光功率/误码门禁**：是否已运行脚本验证全网 `100%` 端口 `LLDP` 邻居与设计 `Cutsheet YAML` 完全一致，且所有光模块 `Pre-FEC BER < 1e-12`、`OCS` 插损 `<1.8 dB`？
- [ ] **阶段 6 黄金固件基线锁死（`Golden Firmware Lock`）**：同一集群内所有节点的 `BMC`、`BIOS`、`NIC/DPU`、交换机 `NOS` 与 `GPU/TPU` 驱动固件版本是否 `100%` 哈希一致？
- [ ] **阶段 7 & 8 `48 小时` 满载烤机（`>=90% busbw / ICI`）与现场备件池就绪**：是否已通过连续 `48h` 满功率热冲击与跨柜集合通信验收（`busbw >= 90%`、参考模型步时方差 `<=2.5%`），并在机房本地库房备妥 `2%–3%` 已预烤合格的光模块、网卡与节点热备件？
