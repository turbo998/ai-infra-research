# AI 服务器 PCIe Gen5 信号完整性、`AER` 可纠正错包重传风暴（`Replay Rollover`）、链路静默降级（`Gen5 x16 -> Gen3 x4`）与 `XID 79`（掉卡 `Fallen Off the Bus`）选型指南

> **Provenance & Evidence Tier**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`)。定量数据锚定于 [`experiments/gpu_pcie_aer_correctable_replay_link_retrain_xid79_selection/results.json`](../../experiments/gpu_pcie_aer_correctable_replay_link_retrain_xid79_selection/results.json)。
> **English Version**: [AI Server PCIe Gen5 Signal Integrity, `AER` Correctable Replay Storms, Silent Link Width/Speed Downgrade & `XID 79` (`Fallen Off the Bus`) Selection Guide](./gpu-pcie-aer-correctable-replay-timer-nak-link-retrain-xid79-selection-faq.en.md)

![AI Server PCIe Gen5 AER Replay & XID 79 Summary](../../assets/svg/gpu-pcie-aer-correctable-replay-timer-nak-link-retrain-xid79-selection-faq-summary.svg)

---

## 0. 🔥 今日领域最热门能力突破追踪（Daily Hottest Breakthrough）

- **通过 `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` 斜率预警（`> 50 次/分`）+ `lspci LnkSta`（`32GT/s x16`）强门禁，在 `Replay Rollover` 演变为致命 `XID 79`（掉卡）或静默退化为 `Gen3 x4`（`3.8 GB/s` 慢卡）之前实现 `100%` 零中断前置隔离**：在 8 卡 H100/H200/B200 服务器中，GPU 与 CPU、以及 GPU 与 400G/800G 网卡通过 **PCIe Gen5 x16（单通道速率高达 `32 GT/s`，高频信号衰减极剧烈）** 互连。当主板 Retimer 芯片虚焊、金手指氧化或机箱高温导致 PCIe 信号完整性劣化时，集群会经历三个致命阶段：
  1. **阶段一（链路层 `AER Correctable Error` 与硬件 `NAK` 重传风暴，Tier 1）**：每当一个 PCIe 数据包（TLP）出现 CRC 校验错误（`Bad TLP / Bad DLLP / Receiver Error`），对端硬件会立即打回一个 `NAK`，触发发送端从硬件 Retry Buffer 重传。很多运维因为 `dmesg` 被 `PCIe Bus Error: severity=Corrected` 刷爆导致磁盘写满，竟然在内核启动参数里加上 **`pci=noaer`** 掩耳盗铃——殊不知这只关掉了报警日志，底层硬件每分钟仍在疯狂重传上万次，把 GPUDirect RDMA 带宽从 `54.2 GB/s` 拖垮至 **`14.2 GB/s`**！
  2. **阶段二（LTSSM 状态机静默降速/降宽保活陷阱，Tier 2）**：当重传频繁触发链路重训练（`LTSSM Recovery`）且部分通道握手失败时，PCIe 硬件为了不断连，会自动把 **`Gen5 x16`（`54.2 GB/s`）静默降级为 `Gen4 x8` 甚至 `Gen3 x4`（仅 `3.8 GB/s`，慢了 `14.26` 倍）**！此时 `nvidia-smi` 里 8 张卡依然都在，任务不报错，但整台千卡集群的速度被这根 `3.8 GB/s` 的残废通道死死拖住！
  3. **阶段三（连续 4 次重传失败触发 `XID 79: GPU has fallen off the bus`，Tier 1/3）**：一旦同一个 TLP 连续 4 次重传全部失败（`Replay Number Rollover`）或触发 `Completion Timeout`，PCIe 链路宣告彻底死亡，内核抛出臭名昭著的 **`XID 79`**（月均引发 **14 次** 千卡任务崩溃），且软重启（`reboot`）完全无效，必须通过 BMC 执行冷断电（`AC Power Cycle`）！
  4. **主动防御闭环（Tier 4）**：实时采集 `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` 一阶导数（`> 50/min`）与 `nvidia-smi -q` / `lspci` 的当前链路速率/宽度（必须等于最大能力 `Gen5 x16`），在发现早期重传爬升或降速时，于下一个 Checkpoint 边界自动无感替换节点，将训练中途 `XID 79` 崩溃压降至 **`0 次/月`**，稳守 **`54.2 GB/s`** 满血带宽！

---

## 1. 📌 核心结论速览（Executive Summary — 3 分钟看懂）

1. **为什么在内核参数里加 `pci=noaer` 是 AI 集群运维最大的“自杀式操作”？**
   - **通俗大白话 + 第一性原理**：`AER（Advanced Error Reporting，高级错误报告）` 只是安装在 PCIe 高速公路旁边的**“电子眼摄像头”**。
   - 当路面塌方（Retimer 芯片过热或 Riser 卡松动）导致运货卡车（TLP 数据包）频频翻车重跑时，摄像头会疯狂往系统日志发警报。如果你配了 `pci=noaer`，就相当于**把摄像头砸了**——警报是不响了，但高速公路上的卡车照样每分钟翻车几万次（硬件层面的 `NAK` 与 `Replay Timer Timeout` 完全由物理芯片控制，根本不受操作系统控制）！结果就是平时变成拖慢全集群的隐形慢节点，撑不住时直接断桥掉卡（`XID 79`）！
2. **为什么发生 `XID 79 (GPU has fallen off the bus)` 后，执行 Linux `reboot` 软重启找不回丢失的 GPU？**
   - 因为 `XID 79` 意味着 GPU 的 PCIe 接口控制器或上游 PCIe Switch 端口进入了硬件级锁死（`Link Down / Surprise Down` 或供电保护锁存）状态。普通的操作系统软重启（Warm Reboot）**不会切断主板 12V/54V 电源轨**，PCIe 状态机依然卡死在故障锁存态。**唯一恢复手段是调用带外 BMC 执行完整冷断电（`ipmitool chassis power cycle`）**，彻底放掉电容余电重置 PCIe 物理层！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Q1 / Q2 / Q3）

### Q1：如何用一行命令在 1 秒内揪出集群里所有“偷偷降速或降宽（例如从 `Gen5 x16` 掉到 `Gen4 x8` 或 `Gen3 x4`）”的残废 GPU 和 RDMA 网卡？
- **答**：
  - **查 GPU**：运行 `nvidia-smi --query-gpu=index,pci.bus_id,pcie.link.gen.max,pcie.link.gen.current,pcie.link.width.max,pcie.link.width.current --format=csv`（注意：必须在 GPU 处于非空闲 P0 状态或跑起负载时检查 `gen.current`，因为空闲节能态下 GPU 会正常降频到 Gen1，但**通道宽度 `width.current` 在任何时候都绝不允许小于 `16`！** 如果 `width.current` 变成了 `8`、`4` 或 `1`，说明物理通道已经断线劣化！）。
  - **查 ConnectX-7/8 RDMA 网卡**：运行 `lspci -vvv -s <BDF> | grep -E "LnkCap:|LnkSta:"`，对比 `LnkCap`（最大能力 `Speed 32GT/s, Width x16`）与 `LnkSta`（当前协商结果）。只要两者不一致，立即拦截该节点！

### Q2：`XID 79` 在爆炸之前，硬件一定会先发出什么预警信号？
- **答**：在 `95%+` 的物理信号劣化场景中，PCIe 链路绝不是一瞬间从 0 错误直接跳到 `XID 79` 的，而是会先经历数小时到数天的 **链路层重传计数（`DCGM_FI_DEV_PCIE_REPLAY_COUNTER`）指数级爬升**！正常健康的 PCIe Gen5 链路每分钟重传次数应为 `0`（或极偶发的个位数）；一旦某张卡的 `PCIE_REPLAY_COUNTER` 增速超过 **`50 ~ 100 次/分钟`**，说明眼图已经闭合，距离连续 4 次重传失败触发 `XID 79` 只剩时间问题！

### Q3：如果既不想让 `AER` 可纠正错误刷爆 `/var/log/messages` 搞挂系统盘，又想保留硬件监控，正确做法是什么？
- **答**：绝对不要用 `pci=noaer`！正确做法是配置内核 PCIe AER 限流（或在 `rsyslog / journald` 中对 `AER: Corrected error received` 配置 `RateLimitIntervalSec`），同时由 **NVIDIA DCGM Exporter** 走硬件计数器通道实时采集 `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` 接入 Prometheus 告警。

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + LTSSM 劣化路径图 + 主动预测隔离时序图）

### 3.1 架构师 AI 服务器 PCIe Gen5 健康监控与 `XID 79` 防护选型决策树

```mermaid
flowchart TD
    A["8 卡 HGX / PCIe AI 服务器运行高强度训练与 GPUDirect RDMA"] --> B{"如何管理 PCIe AER 报错、链路协商与重传计数器？"}
    B -- "粗暴添加 pci=noaer 或放任重传 (Tier 1)" --> C["⛔ 掩耳盗铃! 底层 NAK 重传拖慢 RDMA 至 14.2 GB/s<br/>4 次重传失败直接引爆 XID 79 (月均 14 次掉卡中断)"]
    B -- "仅监控 GPU 是否在线，不校验 LnkSta 速率/宽度 (Tier 2)" --> D["⚠️ LTSSM 重训练静默降级至 Gen3 x4 (3.8 GB/s)<br/>GPU 显示存活却将千卡集群速度拖慢 14 倍!"]
    B -- "事后捕获 XID 79 + BMC 冷断电自愈 (Tier 3)" --> E["✅ 虽能通过 ipmitool power cycle 自动救回掉卡<br/>但无法避免正在运行的千卡训练任务当场崩溃回滚"]
    B -- "DCGM 重传斜率告警 + LnkSta Gen5x16 门禁 (Tier 4)" --> F["🏆 在眼图劣化初期 (>50 replays/min) 提前标记 Cordon<br/>Checkpoint 边界平滑换机 · 月均 XID 79 掉卡降为 0 次!"]
```

### 3.2 PCIe Gen5 物理信号劣化从 `CRC 错包 -> NAK 重传 -> 静默降宽 -> XID 79 掉卡` 的四步雪崩机制

```mermaid
flowchart LR
    subgraph DEGRADE["⛔ Tier 1/2: 未监控的 PCIe 信号劣化雪崩链"]
        direction TB
        S1["1. 物理层眼图闭合 (Retimer 过热 / 金手指氧化)"] --> S2["2. 数据链路层 LCRC 校验失败 -> 发送 NAK 触发硬件 Replay (带宽跌至 14.2 GB/s)"]
        S2 --> S3["3a. LTSSM 重训练屏蔽坏通道 -> 静默降级为 Gen3 x4 (3.8 GB/s 隐形慢卡!)"]
        S2 --> S4["3b. 同一个 TLP 连续 4 次重传失败 (Replay Rollover) -> Fatal Link Down -> XID 79 掉卡崩溃!"]
    end
    subgraph GUARD["🏆 Tier 4: 双重硬件门禁提前拦截"]
        direction TB
        G1["实时监控 DCGM_FI_DEV_PCIE_REPLAY_COUNTER (>50/min)"] --> G2["强校验 LnkSta == Speed 32GT/s, Width x16"]
        G2 --> G3["训练中断 0 次 · 稳守 54.2 GB/s 满速!"]
    end
```

### 3.3 `Tier 4` 基于 `PCIE_REPLAY_COUNTER` 斜率与 `LnkSta` 的零中断换机与 BMC 冷重启时序图

```mermaid
sequenceDiagram
    autonumber
    participant GPU as GPU / PCIe Root Port
    participant DCGM as DCGM Exporter & Node Problem Detector
    participant K8S as 调度器与训练控制器
    participant BMC as 带外 BMC (Redfish / IPMI)

    GPU->>DCGM: 1. 信号开始劣化: PCIE_REPLAY_COUNTER 增速突破 80 次/分钟 (尚未触发 XID 79)
    DCGM->>K8S: 2. 触发 Pre-XID79 亚健康预警，立即将节点标记为 K8s Cordoned (禁止新任务调度)
    K8S->>K8S: 3. 通知当前训练任务在下一个常规 Checkpoint 保存完成后优雅迁出至热备节点 (0 步数损失!)
    K8S->>BMC: 4. 节点腾空后下发 ipmitool chassis power cycle 冷断电重训链路，随后跑 nvbandwidth 复检
```

---

## 4. 📊 架构师选型决策矩阵（4-Tier Selection Matrix）

| 评估维度 | Tier 1: 放任重传或 `pci=noaer` | Tier 2: 漏检 `LTSSM` 静默降宽降速 | Tier 3: 事后 `XID 79` 捕获 + 冷重启 | Tier 4: **DCGM 重传斜率预警 + `LnkSta` 门禁** |
| :--- | :--- | :--- | :--- | :--- |
| **故障节点有效 PCIe 带宽** | `14.2 GB/s` (重传风暴) ✕ | **`3.8 GB/s` (`Gen3 x4` 残废慢卡)** ✕ | `48.5 GB/s` ✅ | **`54.2 GB/s` (`Gen5 x16` 满血)** 🏆 |
| **是否能拦截静默降宽 (`x16->x4`)** | 否 ✕ | 否（变成隐形慢节点） ✕ | 是 ✅ | **是（100% 准入强校验）** 🏆 |
| **月均 `XID 79` 训练崩溃次数** | **`14 次/月`** ✕ | `6 次/月` ⚠️ | `5 次/月` (事后恢复) ⚠️ | **`0 次/月` (事前预测平滑换机)** 🏆 |

---

## 5. 🧪 定量算例与工具链（Interactive Sizer & 7-Card Visual Report）

- **实验脚本与密封数据**：[`experiments/gpu_pcie_aer_correctable_replay_link_retrain_xid79_selection/run.py`](../../experiments/gpu_pcie_aer_correctable_replay_link_retrain_xid79_selection/run.py) · [`results.json`](../../experiments/gpu_pcie_aer_correctable_replay_link_retrain_xid79_selection/results.json)
- **交互式 PCIe AER 重传与 XID 79 评估器**：[`docs/reports/gpu-pcie-aer-correctable-replay-timer-nak-link-retrain-xid79-sizer.html`](../reports/gpu-pcie-aer-correctable-replay-timer-nak-link-retrain-xid79-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：[`docs/reports/gpu-pcie-aer-correctable-replay-timer-nak-link-retrain-xid79-selection-cards.html`](../reports/gpu-pcie-aer-correctable-replay-timer-nak-link-retrain-xid79-selection-cards.html)

---

## 6. 🛠️ Good Fit vs. Not For & 生产环境快速核验清单（Checklist）

### ✅ Good Fit vs. ⛔ Not For

| 维度 | ✅ Good Fit（强烈推荐） | ⛔ Not For（严禁误用） |
| :--- | :--- | :--- |
| **PCIe 错包与掉卡治理** | 采集 `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` 变化率并在任务启动前校验 `LnkSta` 的 `Speed` 与 `Width == x16`；对已发生 `XID 79` 的机器自动触发 BMC `power cycle` | 在 `/etc/default/grub` 中配置 `pci=noaer` 假装无事发生，或对 `XID 79` 机器只做普通 `sudo reboot` 软重启 |

### 生产环境上线核验清单（Pre-Flight Checklist）

- [ ] **核验全集群 `/proc/cmdline` 不含 `pci=noaer`**：若需防止日志撑爆磁盘，改用 `journald` 限流规则而非关闭内核 AER 子系统。
- [ ] **在 Slurm Prolog / K8s InitContainer 中加入 PCIe 通道宽度硬校验**：确认所有 GPU 与 RDMA 网卡的 `pcie.link.width.current == 16`（杜绝 `x8 / x4` 静默残血上线）。
- [ ] **配置 Prometheus 告警规则 `rate(DCGM_FI_DEV_PCIE_REPLAY_COUNTER[5m]) * 60 > 50`**：一旦单卡每分钟重传超 50 次，立即自动标记节点 `Cordon`。
