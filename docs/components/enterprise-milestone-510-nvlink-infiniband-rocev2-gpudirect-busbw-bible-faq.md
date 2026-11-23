# 第 510 篇里程碑：企业智算高速互联网络全景圣经——Scale-Up（`NVLink 4/5 & NVSwitch`）vs. Scale-Out（`InfiniBand NDR/XDR` vs. `RoCEv2 AI 以太网`）、`GPUDirect RDMA` 零拷贝路径、`PFC+ECN+DCQCN` 无损水线、Rail-Optimized 拓扑与 `algbw/busbw` 手算沙盘 (#510)

> **核心观点（Executive Summary）**：
> 在智算中心规划中，网络从来不是附属管道，而是决定千卡算力能否合成为“一台超级计算机”的生命线。作为 **第 510 篇里程碑**，本专题为 **Cloud & AI Solution Architect** 建立一套端到端的智算网络量化选型与手算准则：
> 1. **Scale-Up（`NVLink 4/5 + NVSwitch`）vs. Scale-Out（`InfiniBand vs. RoCEv2`）物理量级鸿沟**：机内 `NVLink 4.0`（`900 GB/s` 双向，`busbw = 371 GB/s`，延迟 `<0.8 us`）与 `NVLink 5.0`（`1,800 GB/s`）比单张 `400G RDMA` 网卡（`50 GB/s` 单向）宽 **`7.7x–18x`**！因此 **高频同步的 `TP` 必须锁死在 Scale-Up 域**，而 Scale-Out 仅承载 `DP / FSDP / PP / EP / PD 分离 KV 传输`；
> 2. **`GPUDirect RDMA` 零拷贝生命线**：若未加载 `nvidia-peermem` 或未关闭主板 `PCIe ACS` 重定向，数据被迫绕行 `GPU HBM -> Host CPU DRAM Bounce Buffer -> RNIC`，400G 网卡实测 `busbw` 从 **`48.2 GB/s` 暴跌至 `11.4 GB/s`（损失 `76.3%` 带宽，慢 `4.23 倍` ✕）**；
> 3. **`InfiniBand NDR/XDR` vs. `AI 以太网 RoCEv2` 选型红线**：通过 **`Rail-Optimized 叶脊拓扑` + `ECN 早于 PFC 标记 (K_min=200KB, K_max=1600KB)` + `动态负载均衡 (DLB/喷洒路由)` + `硬件 PFC Watchdog`**，400G RoCEv2 实测 `AllReduce busbw` 可达 **`46.4 GB/s`（达到 InfiniBand `48.2 GB/s` 的 `96.3%` 水准 🏆）**，同时节省 **`32.5%` 网络交换机与光模块 CapEx**！
>
> **⚡ 配套交互式 H5 精读沙盘**：[**打开第 510 篇里程碑：智算网络 NVLink/IB/RoCEv2 与 algbw/busbw 手算计算器 (`docs/h5/milestone-510-nvlink-ib-rocev2-gpudirect-busbw-lab.html`)**](../h5/milestone-510-nvlink-ib-rocev2-gpudirect-busbw-lab.html)

![Milestone 510 Enterprise AI Fabric Bible Summary](../../assets/svg/enterprise-milestone-510-nvlink-infiniband-rocev2-gpudirect-busbw-bible-faq-summary.svg)

---

## 一、 灵魂拷问：GPUDirect RDMA、无损流控水线、拓扑设计与 `algbw/busbw` 如何精算？

### Q1：什么是 `GPUDirect RDMA` 零拷贝路径？为什么少配一个内核模块或 BIOS 选项会让 400G 网卡退化成 100G？
- **零拷贝 DMA 路径（`GPUDirect RDMA` + `PCIe Switch P2P`）**：
  - 当挂载 `nvidia-peermem` 且关闭 PCIe Switch `ACS (Access Control Services)` 时，同 PCIe Root Complex 下的 `ConnectX-7 400G RNIC` 通过 PCIe BAR1 地址映射直接读写 `GPU HBM`，**全程零 CPU 参与、零主机内存拷贝**，400G 单卡实测 `AllReduce busbw` 达 **`46.4–48.2 GB/s`（线速效率 `92.8%–96.4%`）**，点对点延迟仅 **`1.5–2.6 us`**。
  - **一旦回退至 Host Bounce Buffer（CPU 内存弹跳）**：每次收发都必须经过 `GPU -> PCIe -> CPU Root Complex -> Host Pinned RAM -> RNIC` 两次跨越 PCIe 总线与 NUMA 总线，实测 `busbw` 骤降至 **`11.4 GB/s`**，延迟飙升至 **`18.5 us`**！

### Q2：RoCEv2 以太网如何调优 `PFC + ECN + DCQCN` 水线与 `Rail-Optimized` 拓扑，才能避免 PFC 死锁风暴？
1. **流控先后顺序铁律——“先 ECN 温柔降速，后 PFC 紧急刹车”**：
   - **`ECN + DCQCN`（端到端拥塞控制）**：当交换机出端口队列超过低水线 $K_{\min} = \mathbf{200\text{ KB}}$（400G 端口）时，按概率在 IP 头打上 `CE` 拥塞标记；接收端网卡生成 `CNP (Congestion Notification Packet)` 通知发送端网卡平滑调低发送速率，**在缓冲区满溢前化解 95%+ 的微突发**；
   - **`PFC`（Priority Flow Control 逐跳反压）**：仅当瞬态突发击穿高水线 $X_{\text{off}}$（预留至少 $\text{Headroom} \ge \text{RTT} \times \text{LineRate}$ 飞行字节）时，才向上游发送 `PAUSE` 帧防止丢包；并**强制开启交换机硬件 `PFC Watchdog`（如连续暂停 `>200ms` 自动丢弃隔离异常死锁队列）**。
2. **`Rail-Optimized`（轨道优化拓扑）vs. 传统 `Fat-Tree`**：
   - 将所有服务器同一卡号（如所有机器的 `GPU 0`）汇聚到同一个 `Rail Leaf` 交换机下。机内跨轨走 `900 GB/s NVLink`，跨机同轨只需 **1 跳 Spine/Rail 交换机**，消除传统 ECMP 跨轨哈希极化，并减少 **`25%–50%` 上行 Spine 交换机与光模块数量**。

### Q3：验收集群 running `nccl-tests` 时，`algbw`（算法带宽）与 `busbw`（总线带宽）如何手算换算？
- **算法带宽（$\text{algbw}$）**：张量字节数除以通信耗时 $\text{algbw} = \frac{S_{\text{bytes}}}{t_{\text{sec}}}$；
- **总线带宽（$\text{busbw}$，衡量物理网卡/NVLink 被压榨的真实硬件效率）**：
  $$\text{busbw}_{\text{AllReduce}} = \text{algbw} \times \frac{2(N - 1)}{N}, \qquad \text{busbw}_{\text{AllGather / ReduceScatter / AllToAll}} = \text{algbw} \times \frac{N - 1}{N}$$
  - **手算算例**：在 `N=64` 卡 `400G RoCEv2`（单卡理论线速 `50 GB/s`）集群上跑 `all_reduce_perf`，若测得 `algbw = 23.6 GB/s`，则真实硬件总线带宽为 $\text{busbw} = 23.6 \times \frac{2 \times 63}{64} = 23.6 \times 1.96875 = \mathbf{46.46\text{ GB/s}}$（相当于吃满 400G 物理线速的 **`92.9%` ✓**）！
  - **网内计算（`IB SHARP` / `NVSwitch NVLS`）突破 2 倍系数限制**：交换机芯片在转发途中直接完成浮点数加法归约，将 `2(N-1)/N` 通信量削减近半，使 `400G IB NDR` 的 `algbw` 从 `24.5 GB/s` 直接跃升至 **`34.8 GB/s`（提速 `+42%` 🏆）**！

---

## 二、 企业智算高速互联网络选型决策矩阵（`Scale-Up` & `Scale-Out 400G`）

```mermaid
flowchart TB
    NET["企业智算网络架构分层决策"] --> UP["机内/机柜内 Scale-Up 域<br/>NVLink 4.0 (900 GB/s) / NVLink 5.0 (1.8 TB/s)<br/>承载 TP≤8 与机柜内超低延迟同步"]
    NET --> OUT["跨节点 Scale-Out RDMA 域<br/>承载 DP / FSDP / PP / EP / PD 分离 KV"]
    OUT --> C1["集群规模 ≤512 GPU 或后训练/PD推理共池<br/>首选 400G AI 以太网 RoCEv2 (Rail + ECN/DLB) 🏆<br/>busbw = 46.4 GB/s (达 IB 96.3%) · 省 32.5% CapEx"]
    OUT --> C2["集群规模 ≥1,024 GPU 大规模同步预训练/重度 MoE<br/>首选 400G/800G InfiniBand NDR/XDR + SHARP 🏆<br/>busbw = 48.2 GB/s · SHARP algbw 34.8 GB/s"]
```

| 智算网络方案与配置状态 | 理论单向带宽/卡 | `AllReduce algbw` | `AllReduce busbw` | 点对点延迟 | 物理线速效率 | 相对网络 CapEx |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **反模式 1：`400G RoCEv2` 未开 GPUDirect（CPU Bounce）** | `50.0 GB/s (400G)` | `5.8 GB/s` ✕ | `11.4 GB/s` ✕ | `18.5 us` ✕ | `22.8%` ✕ | `67.5%` |
| **反模式 2：`400G RoCEv2` 静态五元组 ECMP（无 DLB）** | `50.0 GB/s (400G)` | `10.9 GB/s` ✕ | `21.4 GB/s` ✕ | `6.8 us` ✕ | `42.8%` ✕ | `67.5%` |
| **方案 A：`400G AI 以太网 RoCEv2`（`Rail + ECN/DCQCN + DLB` 🏆）** | `50.0 GB/s (400G)` | **`23.6 GB/s` ✓** | **`46.4 GB/s` 🏆** | **`2.6 us` ✓** | **`92.8%` 🏆** | **`67.5% (-32.5%)` 🏆** |
| **方案 B：`400G InfiniBand NDR`（`Credit + AR + SHARP` 🏆）** | `50.0 GB/s (400G)` | **`34.8 GB/s (SHARP)` 🏆** | **`48.2 GB/s` 🏆** | **`1.5 us` 🏆** | **`96.4%` 🏆** | `100.0%` |
| **Scale-Up 基准：机内 `H200 NVLink 4.0 + NVSwitch`** | `450.0 GB/s (双向900G)` | **`212.0 GB/s`** | **`371.0 GB/s`** | **`0.8 us`** | `82.4%` | `主板内置` |

---

## 三、 生产落地核对清单（SA Production Checklist）

- [ ] **Day-0 强制核验 `nvidia-peermem` 与 `PCIe ACS Off`**：执行 `lsmod | grep nvidia_peermem` 并运行 `all_reduce_perf -b 1G -e 4G`，确认 400G 网卡单轨 `busbw >= 45 GB/s`（杜绝 `11.4 GB/s` CPU 弹跳假象）。
- [ ] **RoCEv2 交换机严格对齐 `ECN 早于 PFC` 水线**：400G 端口配置 `DSCP 26 -> Queue 3 (Lossless RDMA)`、`$K_{\min}=200\text{ KB}, K_{\max}=1600\text{ KB}$`、开启 `Dynamic Load Balancing (DLB) / Packet Spraying` 与 **`PFC Watchdog (200ms)`**。
- [ ] **控制面数据流与 RDMA 数据流物理/队列隔离**：将 BGP、Kubernetes 心跳与 Prometheus 监控放入独立的无阻塞高优队列（如 `CS6`），严禁与 RoCEv2 `Queue 3` 共享 PFC 暂停域。

---

## 四、 延伸阅读与配套工具

- **交互式 H5 精学沙盘**：[docs/h5/milestone-510-nvlink-ib-rocev2-gpudirect-busbw-lab.html](../h5/milestone-510-nvlink-ib-rocev2-gpudirect-busbw-lab.html)
- **7-Card 4:5 视觉评测卡片**：[docs/reports/enterprise-milestone-510-nvlink-infiniband-rocev2-gpudirect-busbw-bible-cards.html](../reports/enterprise-milestone-510-nvlink-infiniband-rocev2-gpudirect-busbw-bible-cards.html)
- **交互式 Sizer 测算器**：[docs/reports/enterprise-milestone-510-nvlink-infiniband-rocev2-gpudirect-busbw-bible-sizer.html](../reports/enterprise-milestone-510-nvlink-infiniband-rocev2-gpudirect-busbw-bible-sizer.html)
- **可复现实验模型**：[`experiments/enterprise_milestone_510_nvlink_infiniband_rocev2_gpudirect_busbw_bible/run.py`](../../experiments/enterprise_milestone_510_nvlink_infiniband_rocev2_gpudirect_busbw_bible/run.py)
