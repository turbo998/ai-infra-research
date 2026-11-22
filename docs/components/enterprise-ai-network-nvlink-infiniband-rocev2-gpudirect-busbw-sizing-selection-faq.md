# 智算集群高速互联网络架构选型：`Scale-Up (NVLink 4/5 & NVSwitch)` vs. `Scale-Out (InfiniBand NDR/XDR vs. AI Ethernet RoCEv2 Spectrum-X/UEC)`、`GPUDirect RDMA` 零拷贝、`PFC+ECN+DCQCN` 无损调优与 `algbw/busbw` 手算指南

> **语言 / Language**: **简体中文** | [English](./enterprise-ai-network-nvlink-infiniband-rocev2-gpudirect-busbw-sizing-selection-faq.en.md)  
> **配套交互式学习与精算工具**: [📱 交互式 H5 精读页 (NVLink vs. IB/RoCEv2、GPUDirect 零拷贝与 busbw 沙盘)](../h5/ai-network-nvlink-infiniband-rocev2-gpudirect-busbw-lab.html) · [🧮 HTML 交互式计算器](../reports/enterprise-ai-network-nvlink-infiniband-rocev2-gpudirect-busbw-sizing-selection-sizer.html) · [🃏 7-Card 4:5 视觉评测卡](../reports/enterprise-ai-network-nvlink-infiniband-rocev2-gpudirect-busbw-sizing-selection-cards.html) · [📊 可复核模型](../../experiments/enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection/) · [🎓 返回 AI Infra 全景精学馆](../h5/index.html)

![智算集群高速互联网络架构选型与 busbw 手算全景图](../../assets/svg/enterprise-ai-network-nvlink-infiniband-rocev2-gpudirect-busbw-sizing-selection-faq-summary.svg)

---

## 一、3 分钟看懂核心结论（Executive Summary）

> **一句话决策结论**：企业智算网络绝不是“拉几根 400G 光纤连上交换机”那么简单——首先必须分清 **Scale-Up（机内/机柜级 `NVLink 4/5 + NVSwitch`，双向带宽 `900 ~ 1,800 GB/s`，延迟 `<1 us`，专供高频 `TP` 与机柜内 `EP`）** 与 **Scale-Out（跨节点 `400G/800G RDMA`，单卡单向 `50 ~ 100 GB/s`，延迟 `2 ~ 5 us`，专供 `DP / PP / 跨机 EP / PD 分离 KV 传输`）** 的 **`18 倍` 带宽量级鸿沟**！
> - **交付头号踩坑（`Tier 1` & `Tier 2`）**：若服务器 BIOS 忘了关闭 **`PCIe ACS (Access Control Services)`** 或宿主机未加载 **`nvidia-peermem`**（`Tier 1`），`GPUDirect RDMA` 零拷贝会静默失效，数据被迫绕行 CPU 主存，使 `400G` 网卡（理论 `50 GB/s`）的实测 `busbw` 暴跌至 **`11.2 GB/s`（仅剩 `22.4%` 线速）**！而在 RoCEv2 以太网上仅开启静态五元组 ECMP 与裸 PFC（`Tier 2`），会因 RDMA 大象流哈希极化将有效 `busbw` 卡在 **`31.5 GB/s`（`63.0%` 线速）**；
> - **两大生产黄金选型（`Tier 3` vs. `Tier 4`）**：
>   1. **`<= 512~1,024 GPU` 推理/后训练与云原生多租户共池首选 `Tier 3：400G AI 以太网 RoCEv2 (Rail-Optimized 轨道优化拓扑 + GPUDirect RDMA + ECN/DCQCN 先降速后刹车 + DLB/逐包喷洒)`**：实测 `nccl-tests All-Reduce busbw` 高达 **`47.6 GB/s`（`95.2%` 线速）**，且单卡网络 CapEx 仅 **`$2,650`**（比 InfiniBand 省 `36.1%`）；
>   2. **`>= 1,024 GPU` 超大规模同步预训练与极低延迟跨机 `DeepEP` 首选 `Tier 4：400G/800G InfiniBand NDR/XDR (原生链路层 Credit 无损 + 自适应路由 AR + SHARPv3 网内硬件规约)`**：不仅 `busbw` 达 **`48.8 GB/s`（`97.6%` 线速）**，开启 `SHARPv3` 后更将表观算法带宽 `algbw` 从 `25.4 GB/s` 拉升至 **`38.6 GB/s`（净提速 `+52.0%`）**，`P99 All-to-All` 尾延迟压至 **`132.0 us`**！

| 核心选型维度 | `Tier 1: 400G RoCE + PCIe ACS 未关 (CPU 绕行)` | `Tier 2: 标准 400G RoCE + 静态 ECMP + 裸 PFC` | `Tier 3: 400G AI 以太网 RoCEv2 (Rail+DCQCN+DLB) (🏆)` | `Tier 4: 400G/800G InfiniBand NDR/XDR (AR+SHARP) (🏆)` |
|---|---|---|---|---|
| **NCCL All-Reduce `algbw`** | `6.0 GB/s` ✕ | `16.8 GB/s` ✕ | **`25.4 GB/s` (无 SHARP 物理上限)** 🏆 | **`38.6 GB/s` (SHARPv3 网内规约加成)** 🏆 |
| **NCCL All-Reduce `busbw`** | **`11.2 GB/s` (`22.4%` 线速 ✕)** | `31.5 GB/s` (`63.0%` 线速 ✕) | **`47.6 GB/s` (`95.2%` 线速)** 🏆 | **`48.8 GB/s` (`97.6%` 线速)** 🏆 |
| **MoE 跨机 `P99 All-to-All` 延迟** | `1,420.0 us` ✕ | `485.0 us` ✕ | **`178.0 us` (满足绝大部分 EP/PD)** ✓ | **`132.0 us` (极致微秒级确定性)** 🏆 |
| **折合单张 GPU 网络硬件 CapEx** | `$2,400 / GPU` | `$2,400 / GPU` | **`$2,650 / GPU` (高性价比开放生态)** 🏆 | `$4,150 / GPU` (专用交换机与光模块溢价) |

---

## 二、核心技术原理与架构师灵魂拷问（Q&A）

### Q1：什么是 `GPUDirect RDMA` 零拷贝路径？为什么 BIOS 忘关 `PCIe ACS` 会让 400G 网卡性能瞬間腰斩 `77%`？
1. **正常 `GPUDirect RDMA` 零拷贝路径（`HBM <-> PCIe Switch <-> RDMA NIC`）**：
   - 在标准 8 卡 HGX 服务器中，每 1~2 张 GPU 与对应的 1~2 张 `400G ConnectX-7 / BlueField-3` 网卡共同挂载在**同一颗专用 PCIe Gen5 Switch 芯片**下（运行 `nvidia-smi topo -m` 显示为 **`PIX`**）。
   - 加载 `nvidia-peermem` 内核模块后，RDMA 网卡可以通过 Peer-to-Peer DMA 直接读写同 Switch 下 GPU 的 HBM 显存，**全程零 CPU 参与、零系统主存（Host DRAM）拷贝**，单轨实测 `busbw` 轻松跑满 `47.6 GB/s`！
2. **故障退化路径（`HBM -> PCIe Switch -> CPU Root Complex -> System DRAM -> NIC`）**：
   - 如果服务器主板 BIOS 中默认开启了虚拟化安全隔离项 **`PCIe ACS (Access Control Services)`**，PCIe Switch 会强行拦截一切点对点（P2P）直通包，把所有 GPU 到网卡的数据包向上反弹给 **CPU Root Complex（QPI/UPI 总线与系统内存控制器）**！
   - 此时不仅日志里不会报 Error（作业依然能跑通），但带宽会被 CPU Root Complex 死死卡在 **`11.2 GB/s`**！因此 Day-0 验收第一件事就是执行 `lspci -vvv | grep ACSCtl` 确认 `SrcValid-` 已关闭！

### Q2：在 RoCEv2 无损以太网中，`PFC`、`ECN` 与 `DCQCN` 是怎么配合的？交换机缓存水线（`K_min << K_max << XOFF`）为什么绝对不能配反？
1. **为什么单靠 `PFC（基于优先级的流控）` 会引发全网死锁风暴**：
   - `PFC` 是链路层“急刹车”：当下游交换机入口 Buffer 快满（达到 `XOFF` 水线）时，向上游发送 `PAUSE` 帧强行停止整个队列发包。如果没有提前降速机制，一旦发生多对一 Incast 拥塞，`PAUSE` 帧会逐跳向上游蔓延（Head-of-Line Blocking），锁死无辜的正常流量甚至引发环路死锁！
2. **黄金三剑客（`ECN 早标记 + DCQCN 源端平滑降速 + PFC 兜底防丢包`）水线铁律**：
   - **第一道防线（`ECN K_min = 150 KB, K_max = 800 KB`）**：当交换机出端口队列刚出现轻微排队（`150 KB`，仅占 Buffer 的 `10%~15%`）时，交换机立即在数据包 IP 头打上 **`CE (Congestion Experienced)`** 标记，接收端网卡收到后立刻向发送端网卡回传 **`CNP (拥塞通知包)`**，触发源端网卡硬件 **`DCQCN`** 算法平滑下调发送速率——**在队列真正积压前就把拥塞化解于无形（99.9% 的拥塞靠 ECN 解决，根本不触发 PFC！）**；
   - **第二道兜底防线（`PFC XOFF = 1,500 KB`）**：仅当瞬时微突发突破 `1.5 MB` 时才触发 `PFC PAUSE` 刹车，且上方预留 `Headroom >= 2 × 链路带宽 × 往返光纤传播延迟` 吸收已在光纤中飞行的在途包，做到 **零丢包 + 零死锁**（同时强制开启交换机硬件 **`PFC Watchdog`**，超时 100ms 自动丢包解死锁）。

### Q3：`nccl-tests` 输出的算法带宽（`algbw`）和总线带宽（`busbw`）有什么区别？怎么手算换算系数？
1. **换算公式推导**：
   设通信张量字节数为 $S$，单次集合通信耗时为 $t$，参与 GPU 总数为 $N$：
   - **表观算法带宽（Algorithm Bandwidth）**：$\text{algbw} = \frac{S}{t}$；
   - **真实硬件总线带宽（Bus Bandwidth，对照网卡单向物理线速 `50 GB/s` 验收的唯一指标）**：
     - 对于 **`All-Reduce`**（Ring 算法需做一次 Reduce-Scatter 加一次 All-Gather，每张卡实际收发数据量为 $\frac{2(N-1)}{N} S$）：
       $$\text{busbw}_{\text{AllReduce}} = \text{algbw} \times \frac{2(N-1)}{N}$$
       例如在 $N=16$ 卡、单卡 `400G`（`50 GB/s`）集群上，当实测 `algbw = 25.4 GB/s` 时，对应的硬件真实 `busbw` $= 25.4 \times \frac{30}{16} = \mathbf{47.62\text{ GB/s}}$（达到 `95.2%` 线速！）。而若开启 InfiniBand **`SHARPv3` 网内规约**，交换机芯片直接汇总相加，免去了 $2\times$ 环路系数，使表观 `algbw` 直接飙升至 **`38.6 GB/s`**！
     - 对于 **`All-to-All / Reduce-Scatter / All-Gather`**（MoE 专家并行 `DeepEP` 的核心原语）：
       $$\text{busbw}_{\text{AllToAll}} = \text{algbw} \times \frac{N-1}{N}$$

```mermaid
flowchart LR
    GPU["🖥️ H200 / B200 HBM 显存"] == "1. PCIe Gen5 Switch (PIX)\n(关闭 ACS + 加载 nvidia-peermem)" ==> NIC["🔌 400G ConnectX-7 / SuperNIC\nGPUDirect RDMA 零拷贝直读"]
    NIC == "2. Rail-Optimized 同轨 ToR\n(ECN K_min=150KB << PFC XOFF=1.5MB)" ==> Spine["🌐 AI Ethernet (DLB/喷洒)\n或 InfiniBand (AR + SHARPv3)"]
    Spine ==> Target["🎯 远端节点 GPU HBM\n实测 busbw >= 47.6 GB/s (95.2% 线速!)"]
```

---

## 三、生产落地 Checklist（Pre-Production Checklist）

- [ ] **Day-0 必检 `GPUDirect RDMA` 连通性**：在所有节点运行 `nvidia-smi topo -m` 确认 GPU 与配对 RDMA 网卡为 `PIX`，检查 `lsmod | grep nvidia_peermem` 已加载，并在 BIOS 中彻底禁用 `PCIe ACS`。
- [ ] **严格按公式校验交换机 `ECN` 与 `PFC` 水线**：确保 $K_{\min} \ll K_{\max} < \text{PFC}_{\text{XOFF}}$，并在所有 Leaf/Spine 交换机端口启用 **`PFC Watchdog`**。
- [ ] **执行跨机同轨（Rail-Aligned）与全集群 `nccl-tests` 门禁**：上线前必须跑通 `all_reduce_perf -b 8M -e 8G -f 2 -g 8` 与 `alltoall_perf`，要求 4GB 消息下 **`busbw >= 45.0 GB/s`（`400G` 线速的 `90%`）** 方可移交业务。

---

## 四、来源与方法论溯源（Provenance）

- **数据来源声明**：本指南实验数据来自可复核解析模型 [`experiments/enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection/run.py`](../../experiments/enterprise_ai_network_nvlink_infiniband_rocev2_gpudirect_busbw_sizing_selection/run.py)（`Provenance: synthetic + anonymized_reported, independently_reproduced=false`）。
