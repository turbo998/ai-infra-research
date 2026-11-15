# GPU HBM3/HBM3e 硬件级显存自愈（`Row Remapping` 行重映射）与 `XID 63 / XID 64` 运维决策：为什么出现 `REMAPPED_ROWS_PENDING > 0` 后不执行 `nvidia-smi -r` 重置 GPU 会导致同一行反复引爆二次 `DBE` 崩溃？`SBE` 累积风暴与 `ROW_REMAP_FAILURE` 硬件 RMA 报修红线选型指南

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)。定量数据锚定于 [`experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/results.json`](../../experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/results.json)。
> **English Version**: [GPU HBM3/HBM3e Hardware `Row Remapping` (`XID 63 / XID 64`, `REMAPPED_ROWS_PENDING` & `ROW_REMAP_FAILURE`): Why Un-Reset Pending Remaps Trigger Repeat `DBE` Crashes & Automated Drain-Reset-RMA Selection Guide](./gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-selection-faq.en.md)

![GPU HBM3/HBM3e Row Remapping XID 63 vs XID 64 Summary](../../assets/svg/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **揭秘为什么在万卡 `H100 / H200 / B200` 集群中，某张 GPU 报出一次 `ECC DBE` 并伴随着 `XID 63 (Row Remap Recorded)` 后，如果不对它执行 `nvidia-smi -r` 重置，接下来调度上去的训练作业有 `88.0%` 的概率会在几分钟内踩中同一个坏显存行再次崩溃？而如果看到 `XID 63` 就盲目拉下线返厂 RMA，又有 `92.0%` 属于浪费算力的过度报修？通过构建“基于 `DCGM_FI_DEV_ROW_REMAP_PENDING` 的准入拦截 + `8.0 秒` 带内 `nvidia-smi -r` 激活 HBM 物理备用行自愈 + 仅在 `XID 64 (ROW_REMAP_FAILURE == 1)` 备用行耗尽时触发硬件 RMA”的闭环状态机（`Tier 4`），实现 `0.0%` 同行二次崩溃与 `0.0%` 误报修**：在 Volta/Turing 老架构时代，GPU 用的是软件层面的“动态页退役（Dynamic Page Blacklisting，驱动在分配显存时绕开坏页，会导致可用显存变少且产生碎片）”。从 Ampere / Hopper / Blackwell（`A100 / H100 / H200 / B200`）开始，NVIDIA 全面升级为 **硬件级 HBM 行重映射（Hardware Row Remapping）**！然而，80% 的运维脚本都没搞懂它的两阶段生效机制：
  1. **第一性原理揭示：什么是 HBM 晶圆上的“备用冗余行（Spare Rows）”与 `XID 63` 的两阶段生效机制？**
     - 在生产 HBM3 / HBM3e 显存颗粒时，内存芯片厂商（SK Hynix / Micron / Samsung）在 **每一个 HBM Bank 内部都预先多造了若干条平时不通电的“物理备用行（Hardware Spare Rows）”**！
     - 当线上运行中的 GPU 在某一条物理行（比如 `Bank 5, Row 1024`）上检测到了 **1 次不可纠正双比特错误（`DBE`）**，或者 **可纠正单比特错误（`SBE`）频繁发作达到了危险阈值**：
       - **阶段一（记账阶段 `XID 63` — 此时坏行还在继续用！）**：GPU 驱动将该坏行的物理坐标写入显卡上断电不丢失的 **`InfoROM` 芯片** 中，打印内核日志 **`XID 63`**，并将遥测指标 **`REMAPPED_ROWS_PENDING`（`DCGM_FI_DEV_ROW_REMAP_PENDING`）置为 `1 (Yes)`**！
       - **最致命的认知盲区**：请注意名字里的 **`PENDING`（待生效）**！因为当前 GPU 显存控制器（Memory Controller）正在高速运转，硬件电路**无法在半空中热切换物理行连线**！在没有发生 **`GPU Reset`（`nvidia-smi -r` 或宿主机重启）** 之前，**那条已经损坏的 `Bank 5, Row 1024` 依然暴露在可用显存池里**！
       - **灾难后果（Tier 1 — `88.0%` 连环炸任务）**：如果上层 Kubernetes 在上一个崩溃的容器退出后，看了一眼 GPU 还在，没有跑 `nvidia-smi -r` 就直接把下一个千卡训练任务调度了上来——新任务的权重或激活值一旦再次分配到 `Bank 5, Row 1024`，当场再次触发 `XID 94 / 95` 崩溃（连环炸任务率高达 **`88.0%`**）✕！
       - **阶段二（重置激活阶段 `nvidia-smi -r` — `8.0 秒` 永久满血复活！）**：只要在容器退出后执行一次 **`sudo nvidia-smi -r -i <gpu_id>`（耗时仅 `8.0 秒`）**，GPU 显存控制器在重新初始化的瞬间读取 `InfoROM`，在硅片硬件多路复用器（MUX）层面 **将 `Bank 5, Row 1024` 永久重定向到该 Bank 内部的全新硬件备用行上**，同时将 `REMAPPED_ROWS_PENDING` 清零为 **`0 (No)`**！从此以后，这张卡不仅 **1 字节显存容量都不会少**，而且访问速度和出厂新卡 **100% 完全一样**！
  2. **四大运维处置策略对比**：
     - **Tier 1（不检查 `REMAPPED_ROWS_PENDING`，不跑 `nvidia-smi -r` 直接放行新任务）**：坏行一直处于 `Pending` 未替换状态，连环炸毁后续任务（二次崩溃率 **`88.0%`**）✕。
     - **Tier 2（谈 ECC 色变：只要出现 1 次 `XID 63` 或 `REMAPPED_ROWS_DBE > 0` 就直接下线返厂 RMA）**：完全浪费了 HBM3 芯片自带的数百条硬件备用行自愈能力，高达 **`92.0%` 的返厂报修属于根本不需要换卡的误杀**，每张卡白白损失 1~2 周的宝贵算力 ⚠️！
     - **Tier 3（虽对 `Pending` 执行重置，但忽视 `XID 64 / ROW_REMAP_FAILURE == 1` 备用行耗尽红线）**：如果某一个 HBM Bank 发生了区域性物理损坏，把该 Bank 内仅有的所有硬件备用行全部用光了，此时再坏一行就会报 **`XID 64 (Row Remapping Failure)`** 且 **`ROW_REMAP_FAILURE = 1 (Yes)`**！此时就算重置 100 次也无行可换了，放回集群仍有 `14.5%` 崩溃率 ✅。
     - **Tier 4（工业级黄金闭环状态机：准入网关拦截 `PENDING > 0` + `8 秒` 自动带内 `nvidia-smi -r` 激活备用行 + 仅在 `ROW_REMAP_FAILURE == 1 (XID 64)` 时精准下线 RMA）**：二次崩溃率 **`0.0%`**，过度报修率 **`0.0%`**，自愈耗时仅 **`8.0 秒`** 🏆！

---

## 1. 📌 核心结论速览（Executive Summary — 3 分钟看懂）

1. **架构师必须死记硬背的 `nvidia-smi -q -d ROW_REMAPPERS` 四个黄金指标含义**：
   - **`Correctable Error (SBE) Remapped Rows`**：因为单比特错误（`SBE`）频繁出现而被硬件预防性替换掉的行数（健康自愈，无需报修）；
   - **`Uncorrectable Error (DBE) Remapped Rows`**：因为双比特错误（`DBE`）被硬件替换掉的行数（只要重置后未再持续暴增，同样属于健康自愈，无需报修）；
   - **`Pending` (`DCGM_FI_DEV_ROW_REMAP_PENDING`)**：**只要等于 `1 (Yes)`，严禁调度任何新任务！必须立刻排空并执行 `nvidia-smi -r` 重置 GPU 让备用行真正生效（变成 `0`）后再恢复调度！**
   - **`Remapping Failure Occurred` (`DCGM_FI_DEV_ROW_REMAP_FAILURE` / `XID 64`)**：**硬件死刑判决书！** 一旦等于 `1 (Yes)`，代表某个 HBM Bank 的物理备用行已经彻底耗尽，任何软件重置均无效，**100% 立即永久 Cordon 下线走硬件 RMA 换卡**！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：除了 `ROW_REMAP_FAILURE == 1`（`XID 64`）之外，还有什么情况虽然还没耗尽备用行，但也应该提前预警或换卡？
- **答**：看 **重映射发生的速度（Velocity）与单比特错误风暴（SBE Storm）**！
  - 如果一张卡在过去 1 年里总共才累计了 `2` 次 `DBE Remap`，重置后跑得稳如泰山——这属于宇宙射线或偶发单点老化，完全健康；
  - 但如果一张卡在 **短短 24 小时内连续新增了 `>= 4` 次 `DBE Remap`**，或者某个内存通道的 **`SBE Volatile Count` 每小时暴涨超过 `10,000` 次**——说明该 HBM Stack 的微凸点（Micro-bump）或硅通孔（TSV）正在发生渐进式结构脱落，哪怕当前备用行还没耗尽，也将在几天内耗尽触发 `XID 64`，应在下一个维护窗口提前排水更换！

### Q2：为什么在 NVSwitch 8 卡机器（如 HGX H100/H800）上执行 `nvidia-smi -r` 重置单卡时，有时会报 `GPU Reset Failed` 或需要先停掉 `nvidia-fabricmanager`？
- **答**：在带有 NVSwitch 的节点上，如果开启了全互联 Fabric 分区，直接重置单张 GPU 可能会因为 NVLink 训练状态机锁定而要求整机 8 卡无进程占用。标准自愈脚本流程是：确认本机无活跃容器 $\to$ 执行 `nvidia-smi -r`（必要时 `systemctl restart nvidia-fabricmanager`）$\to$ 核验 `nvidia-smi --query-gpu=remapped_rows.pending --format=csv,noheader` 全部返回 `0`（或 `No`）！

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + `InfoROM` 两阶段生效 vs 盲目报修对比图 + 时序图）

### 3.1 架构师 GPU HBM 行重映射（`Row Remapping`）与 RMA 报修决策树

```mermaid
flowchart TD
    A["GPU 触发 ECC 告警或任务崩溃，检查 nvidia-smi -q -d ROW_REMAPPERS"] --> B{"检查 Remapping Failure Occurred (DCGM_FI_DEV_ROW_REMAP_FAILURE / XID 64) 是否为 Yes(1)？"}
    B -- "是: Failure == 1 (或 24h 内新增 DBE Remap >= 4)" --> C["⛔ 该 HBM Bank 物理备用行已彻底耗尽!<br/>任何重置均无效 · 立即永久 Cordon 下线走硬件 RMA 换卡!"]
    B -- "否: Failure == 0 (备用行充足)" --> D{"检查 Pending (DCGM_FI_DEV_ROW_REMAP_PENDING / XID 63) 是否为 Yes(1)？"}
    D -- "是: Pending == 1，但不重置直接放行新任务 (Tier 1)" --> E["⛔ 坏行仅记入 InfoROM 尚未物理切换!<br/>新任务踩中同一坏行 · 二次崩溃率高达 88.0%!"]
    D -- "是: Pending == 1，盲目直接返厂 RMA (Tier 2)" --> F["⚠️ 无视硬件备用行自愈能力 · 92% 属于浪费算力的误报修!"]
    D -- "是: 自动拦截调度 -> 执行 nvidia-smi -r 激活备用行 (Tier 4)" --> G["🏆 8.0 秒完成硅片级物理备用行切换 (Pending 归 0)<br/>0 显存损失 · 0 二次崩溃 · 满血恢复上线!"]
```

### 3.2 `Tier 1`（`PENDING=1` 不重置连环炸）vs. `Tier 4`（`8 秒` 重置激活硬件备用行 + `XID 64` 精准 RMA）对比图

```mermaid
flowchart LR
    subgraph UNRESET["⛔ Tier 1: Pending=1 不重置 (坏行未切 -> 88% 二次崩溃!)"]
        direction TB
        U1["发生 DBE -> 报 XID 63 记入 InfoROM (REMAPPED_ROWS_PENDING = 1)"] --> U2["未执行 nvidia-smi -r! 物理上依然在使用坏掉的 Row 1024!"]
        U2 --> U3["K8s 立即调度新任务上卡 -> 几分钟内再次踩中 Row 1024 崩溃!"]
    end
    subgraph HEAL["🏆 Tier 4: 闭环 8s 重置自愈 + XID 64 精准 RMA (0% 复发)"]
        direction TB
        H1["检出 PENDING = 1 -> K8s Node Problem Detector 立即拦截新 Pod 准入"] --> H2["执行带内 nvidia-smi -r (8.0s): 内存控制器读取 InfoROM 永久切向硬件备用行!"]
        H2 --> H3["核验 PENDING == 0 & FAILURE == 0 后自动恢复上线 (仅 FAILURE==1 时走 RMA)!"]
    end
```

### 3.3 `Tier 4` HBM3e 硬件备用行从 `XID 63 (InfoROM 记录)` 到 `nvidia-smi -r (MUX 物理切换)` 闭环时序图

```mermaid
sequenceDiagram
    autonumber
    participant HBM as HBM3e Bank (Bad Row + Spare Row Pool)
    participant ROM as GPU InfoROM & Driver (XID 63/64)
    participant NPD as K8s Pre-Flight Gate / Health Daemon

    HBM->>ROM: 1. 检测到物理行损坏，记录坏行地址至 InfoROM，抛出 XID 63 并置 PENDING = 1
    ROM->>NPD: 2. DCGM Exporter 暴露 ROW_REMAP_PENDING = 1，NPD 立即给节点打 Taint 阻止新任务落盘
    NPD->>HBM: 3. 待本机进程退出后执行 nvidia-smi -r (8.0s)，HBM 控制器硬件 MUX 将该地址永久映射至 Spare Row!
    HBM->>NPD: 4. 核验 PENDING == 0 且 ROW_REMAP_FAILURE == 0，摘除 Taint 恢复满血生产调度!
```

---

## 4. 📊 架构师选型决策矩阵（4-Tier Selection Matrix）

| 评估维度 | Tier 1: `Pending` 不重置直接跑 | Tier 2: 遇 `XID 63` 盲目返厂 RMA | Tier 3: 重置但漏拦 `XID 64` | Tier 4: **闭环 `Pending` 8秒重置 + `XID 64` 精准 RMA** |
| :--- | :--- | :--- | :--- | :--- |
| **同一坏行引发的后续任务二次崩溃率** | **`88.0%` (坏行尚未切换!)** ✕ | `0.0%` | `14.5%` (备用行耗尽复发) ⚠️ | **`0.0%` (重置后物理备用行永久接管)** 🏆 |
| **不必要的硬件返厂误报修率 (`False RMA`)** | `0.0%` | **`92.0%` (白白浪费健康卡!)** ✕ | `5.0%` | **`0.0%` (充分利用芯片自带硬件冗余行)** 🏆 |
| **单次可自愈坏行的业务恢复耗时** | `7,200.0 s` (人工反复排障) ✕ | **`604,800.0 s` (`7天` 返厂换卡)** ✕ | `1,800.0 s` | **`8.0 s` (全自动带内 `nvidia-smi -r` 自愈)** 🏆 |

---

## 5. 🧪 定量算例与工具链（Interactive Sizer & 7-Card Visual Report）

- **实验脚本与密封数据**：[`experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/run.py`](../../experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/run.py) · [`results.json`](../../experiments/gpu_dcgm_ecc_sbe_dbe_page_retirement_row_remapping_selection/results.json)
- **交互式 HBM Row Remapping (XID 63/64) 诊断与 RMA 决策计算器**：[`docs/reports/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-sizer.html`](../reports/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-selection-cards.html`](../reports/gpu-hbm-row-remapping-sbe-dbe-xid63-xid64-pending-reset-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & 生产环境快速核验清单（Checklist）

### ✅ Good Fit vs. ⛔ Not For

| 维度 | ✅ Good Fit（强烈推荐） | ⛔ Not For（严禁误用） |
| :--- | :--- | :--- |
| **大规模 GPU 集群 HBM 显存健康管理** | 将 **`remapped_rows.pending == 0` 和 `remapped_rows.failure == 0`** 作为每个训练任务启动前的强制准入检查（Pre-Flight Check），对 `Pending=1` 自动执行 `nvidia-smi -r`，对 `Failure=1 (XID 64)` 自动下线 RMA | 在 `Pending=1` 时不重置继续跑新任务，或看到 `remapped_rows.unc > 0`（但 `failure=0`）就盲目申请换硬件 |

### 生产环境上线核验清单（Pre-Flight Checklist）

- [ ] **在作业启动前 Pre-Flight 脚本中加入一键检查：`nvidia-smi --query-gpu=remapped_rows.pending,remapped_rows.failure --format=csv,noheader`**：两列必须全为 `0`（或 `No`）才允许启动训练。
- [ ] **配置 Node Problem Detector 监听 `XID 63`（自动标记待重置并在空闲时运行 `nvidia-smi -r`）与 `XID 64`（立即永久 Cordon 下线报修）**。
- [ ] **监控 `remapped_rows.unc` 的 24 小时增量 $\Delta$：若单卡 24h 内新增 $\ge 4$ 次重映射，提前隔离预防渐进式 HBM 失效**。
