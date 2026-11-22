# 企业 8 卡 GPU 服务器 CPU/NUMA 拓扑与 PCIe 根复合体调优（`关闭 PCIe ACS 陷阱` + `single-numa-node` + `IRQ/Tokenizer 核绑定`）选型指南 (#506)

> **核心观点（Executive Summary）**：
> 在企业智算中心交付中，最令人痛心的浪费是：**花了 `$300,000` 采购顶配 8 卡 `HGX H200 / B200` 服务器，却因为保持服务器出厂默认的 BIOS 与 Linux 调度设置，白白丢掉了 `28%` 的整机吞吐（相当于每买 4 台服务器就有 1 台在空转）**！罪魁祸首是三大主机侧隐形杀手：
> 1. **BIOS 默认开启 `PCIe ACS（Access Control Services）`**：导致挂在同一个 PCIe Switch 下的 `400G 网卡` 与 `GPU` 无法直连通信，所有 GPUDirect RDMA 数据包被迫绕道上行进 CPU Root Complex，**跨机 RDMA 带宽从 `47.8 GB/s` 暴跌 `70.3%` 至 `14.2 GB/s` ✕**！
> 2. **跨 Socket NUMA 内存与线程横跳**：`GPU 0–3` 物理连接在 `CPU Socket 0 (NUMA 0)`，但 Linux 默认调度器把对应的 vLLM Python 主进程和 Pinned DRAM KV 缓存分到了 `Socket 1 (NUMA 1)`，每次 PCIe DMA 都要跨 `UPI / xGMI` 互联总线，导致 **`P99 TPOT` 暴涨至 `46.5 ms` ✕**！
>
> 作为 **Cloud & AI Solution Architect**，无需多花一分钱硬件预算，只需执行 **“关闭 PCIe ACS（`setpci` 清零重定向位）+ 内核开启 `iommu=pt` + K8s 开启 `TopologyManager: single-numa-node` + 网卡软中断（`IRQ`）与异步 Tokenizer 专属绑核隔离”**：即可将 GPUDirect RDMA 带宽恢复 **3.37 倍至 `47.8 GB/s` 满线速**，`P99 TPOT` 压降 **`61.7%` 至 `17.8 ms`**，100% 释放整机硬件极限算力（**净增 `+38.9%` 有效吞吐 🏆**）！
>
> **⚡ 配套交互式 H5 精读沙盘**：[**打开 8 卡 GPU 服务器 PCIe ACS / NUMA 拓扑调优与算力找回计算器 (`docs/h5/gpu-server-numa-pcie-acs-irq-tuning-lab.html`)**](../h5/gpu-server-numa-pcie-acs-irq-tuning-lab.html)

![Enterprise 8-GPU Server NUMA PCIe ACS & IRQ Pinning Summary](../../assets/svg/enterprise-gpu-server-numa-pcie-acs-irq-pinning-tuning-selection-faq-summary.svg)

---

## 一、 灵魂拷问：为什么一条 `setpci` 命令能让跨机 RDMA 带宽瞬间翻 3.37 倍？

### Q1：什么是 `PCIe ACS（访问控制服务）`？为什么服务器厂商出厂默认会打开它？
1. **原因**：通用服务器厂商为了兼容传统虚拟机（KVM VFIO 安全隔离，要求所有 PCIe 点对点流量必须先上送到 CPU IOMMU 检查权限），出厂 BIOS 默认将所有 PCIe Switch 的 `ACS P2P Redirect` 设为 `Enabled`！
2. **后果**：在 8 卡 GPU 服务器里，原本走 PCIe Switch 内部 `PIX` 极速路径（`< 150ns`、`47.8 GB/s`）的 `NIC <-> GPU` 零拷贝流量，全部被强行劫持上送到 CPU Root Complex（`NODE / SYS` 路径），把 CPU PCIe 根端口彻底堵死（跌至 **`14.2 GB/s` ✕**）！
3. **一行命令秒修**：在宿主机开机 DaemonSet 中运行 `lspci` 扫描所有 PCIe Bridge 并执行 `setpci -v -d *:* ECAP_ACS+0x6.w=0000`，配合内核启动参数 `iommu=pt`（Pass-Through），瞬间恢复 **`47.8 GB/s` 满速直连**！

---

## 二、 四档宿主机 BIOS / NUMA / 绑核调优状态选型决策矩阵（`8x H200 服务器`）

```mermaid
flowchart LR
    T1["Tier 1: 出厂默认 BIOS<br/>PCIe ACS 开启 · RDMA 仅 14.2 GB/s<br/>整机吞吐仅释放 72.0% ✕"] --> T3["Tier 3: 关闭 ACS + single-numa-node<br/>消除跨 Socket UPI 绕行<br/>RDMA 47.2 GB/s · 吞吐 95.2% ✓"]
    T3 --> T4["Tier 4: 叠加 IRQ/Tokenizer 绑核隔离 🏆<br/>RDMA 47.8 GB/s · P99 TPOT 17.8ms<br/>100% 满血释放 (+38.9% 净增吞吐 🏆)"]
```

| 8 卡 GPU 服务器主机调优档位 | GPUDirect RDMA 带宽 (`GB/s`) | 主机 DMA 锁页带宽 (`GB/s`) | `P99 TPOT` 吐字延迟 | 整机吞吐释放率 (%) | 架构师推荐度 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: 出厂默认（`PCIe ACS 开启` + 随机 NUMA）** | `14.2 GB/s` ✕ | `24.5 GB/s` | `46.5 ms` ✕ | `72.0%` ✕ | ⛔ 白白损失 28% 昂贵算力 |
| **Tier 2: 仅关闭 `PCIe ACS`（未绑定 NUMA 亲和）** | `44.5 GB/s` | `32.0 GB/s` | `31.0 ms` ✕ | `86.5%` | ⚠️ 跨 Socket 内存访问抖动 |
| **Tier 3: 关闭 `ACS` + K8s `single-numa-node` 对齐** | `47.2 GB/s` ✓ | `54.0 GB/s` ✓ | `21.5 ms` ✓ | `95.2%` ✓ | ✅ 生产合格基线 |
| **Tier 4: 关闭 `ACS` + `NUMA 对齐` + `IRQ/Tokenizer 绑核` 🏆** | **`47.8 GB/s (3.37x)` 🏆** | **`58.5 GB/s` 🏆** | **`17.8 ms (-61.7%)` 🏆** | **`100.0% (+38.9%)` 🏆** | **✅ 零成本满血调优金标准** |

---

## 三、 生产落地核对清单（SA Production Checklist）

- [ ] **双路 AMD EPYC / Intel Xeon 智算节点 BIOS 统一设为 `NPS1 / SNC Disabled`（或严格按 `NPS2` 对齐双 PCIe Root）**：关闭 `CPU C-States` 节能休眠，将 CPU 调频策略锁死为 `performance`。
- [ ] **Kubelet 强制开启 `--topology-manager-policy=single-numa-node` 与 `--cpu-manager-policy=static`**：确保申请 `GPU 0–3` 的推理容器，其分配到的独占 CPU 核心、内存页与网卡 VF 100% 位于 `NUMA Node 0`。

---

## 四、 延伸阅读与配套工具

- **交互式 H5 精学沙盘**：[docs/h5/gpu-server-numa-pcie-acs-irq-tuning-lab.html](../h5/gpu-server-numa-pcie-acs-irq-tuning-lab.html)
- **7-Card 4:5 视觉评测卡片**：[docs/reports/enterprise-gpu-server-numa-pcie-acs-irq-pinning-tuning-selection-cards.html](../reports/enterprise-gpu-server-numa-pcie-acs-irq-pinning-tuning-selection-cards.html)
- **交互式 Sizer 测算器**：[docs/reports/enterprise-gpu-server-numa-pcie-acs-irq-pinning-tuning-selection-sizer.html](../reports/enterprise-gpu-server-numa-pcie-acs-irq-pinning-tuning-selection-sizer.html)
- **可复现实验模型**：[`experiments/enterprise_gpu_server_numa_pcie_acs_irq_pinning_tuning_selection/run.py`](../../experiments/enterprise_gpu_server_numa_pcie_acs_irq_pinning_tuning_selection/run.py)
