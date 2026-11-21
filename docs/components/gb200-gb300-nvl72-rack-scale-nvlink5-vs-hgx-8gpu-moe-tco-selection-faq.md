# 超节点与机柜级智算架构选型：`NVIDIA GB200 / GB300 NVL72`（72 卡全互联 NVLink 5 铜缆背板 + Grace `NVLink-C2C` 一致性内存）vs. 8 卡 `HGX H200 / B200` 推理吞吐、MoE 边界与 TCO 指南

> **语言 / Language**: **中文 (当前)** | [English](./gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与智算中心规划负责人的机柜级超节点（`GB200 / GB300 NVL72`）vs. 传统 8 卡服务器（`HGX H100 / H200 / B200`）Scale-Up 域扩展原理、Grace-Blackwell `NVLink-C2C` 内存池、MoE `EP=72` 吞吐跃迁与机房液冷选型白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业智算硬件选型与容量规划）

![超节点 GB200/GB300 NVL72 vs 8卡 HGX H200/B200 选型总结图](../../assets/svg/gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**传统 8 卡 `HGX` 服务器有一道难以逾越的“8 卡 NVLink 墙”——机内 8 张卡通过 `900 ~ 1,800 GB/s` NVLink 飞速通信，但一旦模型切分超过 8 张卡（如 `DeepSeek-R1 671B` 或万亿 MoE 跑 `EP=64/72`），跨机通信立刻跌落到 `50 ~ 100 GB/s` 的 RDMA 网卡（速度暴跌 `18 倍`）！`NVIDIA GB200 / GB300 NVL72` 通过 18 台 NVSwitch 托盘与 5,000+ 根无源高速铜缆背板，将整柜 **72 张 Blackwell GPU 融合为单一 `130 TB/s`（每卡 `1,800 GB/s`）的 Scale-Up NVLink 5 全互联域**，使 `EP=72` 专家并行 100% 在微秒级 NVLink 内部完成、零跨网 RDMA 开销；叠加 `Grace CPU <-> Blackwell GPU` 之间 **`450 GB/s/GPU` 的 `NVLink-C2C` 一致性内存通道**（比传统 x86 PCIe Gen5 `64 GB/s` 快 **`7 倍`**），单卡达标吞吐飙升至 `7,420 tok/s/GPU`（是 `HGX H200` 的 `3.12 倍`）、P99 TPOT 压至 **`5.2ms`（单用户每秒吐字 `192 tok/s`！）**，百万 Token 成本降至 **`$0.13/1M`**！**

- **交互式 `GB200 NVL72` vs. `8-GPU HGX` 性能与成本对比计算器**：[打开交互式计算器 (`gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-sizer.html`)](../reports/gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-selection-cards.html`)](../reports/gb200-gb300-nvl72-rack-scale-nvlink5-vs-hgx-8gpu-moe-tco-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：同样是 72 张 Blackwell GPU，为什么 1 柜 `GB200 NVL72` 跑大型 MoE（`EP=72`）比 9 台 8 卡 `HGX B200` 快 `60%` 以上？
1. **Scale-Up 域半径扩大 9 倍（8 卡 $\to$ 72 卡全互联）**：
   - 在 9 台独立的 8 卡 `HGX B200` 上跑 `EP=72` 时，只有 `1/9` 的专家流量在机内 `1,800 GB/s` NVLink 5 上完成，其余 **`8/9` 的 `All-to-All` 流量必须走每卡 `100 GB/s`（800G）的跨机 RDMA 网卡**，带宽缩水 **18 倍**！
   - 而在 1 柜 **`GB200 NVL72`**（含 18 个 Compute Trays × 4 GPUs + 9 个 NVSwitch Trays）中，全部 **72 张 GPU 通过机柜背部无源铜缆背板（Copper Backplane）直连在同一个 `1,800 GB/s/GPU` NVLink 5 交换域内**！跑 `EP=72` 或 `TP=16/32` 时，100% 流量走 `<1 us` 的 NVLink 5 + `NVLS`，完全不需要经过网卡，不仅省掉 72 个昂贵的高功耗光模块（省电 `20+ kW`），还把 P99 TPOT 直接压到 **`5.2 ms`**！
2. **Grace CPU `NVLink-C2C` 打破 x86 PCIe 内存墙**：
   - 传统 x86 服务器中，每张 GPU 通过 PCIe Gen5 x16 连接 CPU 主存，单向带宽仅 **`64 GB/s`**；
   - `GB200 Superchip` 将 1 颗 Grace CPU（配 `480 GB LPDDR5X`）与 2 颗 B200 GPU 通过 **`900 GB/s NVLink-C2C`（即每张 GPU 独享 `450 GB/s`，是 PCIe Gen5 的 `7.0 倍`！）** 硬件缓存一致性互联，使整柜 `17 TB` 的 CPU 内存变成毫秒级直读的超大 KV Cache 与 RL 权重卸载池！

### Q2：既然 `GB200 NVL72` 这么强，什么场景下我们仍然应该选 8 卡 `HGX H200 / B200`？
- **机房基建硬约束与工作负载匹配原则**：
  - **选 `8-GPU HGX H200 / B200` 的三大场景**：① 存量机房单机柜供电/制冷上限 `<= 40 ~ 60 kW`（不具备 `120 ~ 132 kW/柜` 高密冷板液冷 + CDU 改造条件）；② 主力业务是 `<= 70B Dense` 模型（单实例 `TP=2/4/8` 完全装在 1 台 8 卡服务器内，根本用不到跨 72 卡的 NVLink）；③ 采购规模小于 72 张卡的中小企业。
  - **选 `GB200 / GB300 NVL72` 的核心场景**：新建 `>= 120 kW/柜` 原生液冷智算中心，且主力承载 `DeepSeek-671B` / 万亿参数 MoE 极速推理、深度思考链（`<think>` 需 `150~200 tok/s` 单流极速）与超大规模前沿预训练。

---

## 3. 🧭 直观可视化图解（Mermaid 9×HGX vs. 1×NVL72 架构对比）

### 3.1 `9 台 8卡 HGX`（跨机 RDMA 瓶颈）vs. `1 柜 GB200 NVL72`（72卡全铜缆 NVLink 5）架构对比
```mermaid
flowchart TD
    subgraph HGX["传统架构: 9 台 8卡 HGX H200/B200 跑 EP=72"]
        H1["机内 8 卡 NVLink (900~1,800 GB/s)"] --> H2["⚠️ 跨节点 8/9 的 MoE All-to-All 流量<br/>被迫走 50~100 GB/s RDMA 网卡 (降速 18x!)"]
        H2 --> H3["x86 PCIe Gen5 主存卸载仅 64 GB/s/GPU"]
    end
    subgraph NVL["🏆 超节点架构: 1 柜 GB200 / GB300 NVL72 跑 EP=72"]
        N1["18 个 Compute Trays (72 GPUs) + 9 个 NVSwitch Trays"] --> N2["全柜 72 卡共享 130 TB/s 无源铜缆 NVLink 5 背板!<br/>• EP=72 100% 走 1,800 GB/s 机柜内 NVLink (<1us)"]
        N2 --> N3["36 颗 Grace CPU 提供 17TB LPDDR5X (450 GB/s/GPU C2C!)"]
    end
```

---

## 4. 📊 架构师多方案选型决策矩阵（72 卡规模四代硬件底座对比）

| 评估维度 | Tier 1：9 台 `8x H100 HGX`（400G RDMA） | Tier 2：9 台 `8x H200 HGX`（141GB + `DeepEP`） | Tier 3：9 台 `8x B200 x86 HGX`（800G RDMA） | Tier 4（推荐超节点）：1 柜 `GB200 / GB300 NVL72`（72卡 NVLink 5 + C2C） |
| :--- | :--- | :--- | :--- | :--- |
| **单 Scale-Up NVLink 域最大卡数** | `8 张卡` (`900 GB/s`) | `8 张卡` (`900 GB/s`) | `8 张卡` (`1,800 GB/s`) | **`72 张卡` (`1,800 GB/s`，合计 `130 TB/s`!)** |
| **CPU<->GPU 内存卸载带宽/卡** | `64 GB/s` (PCIe Gen5) | `64 GB/s` (PCIe Gen5) | `64 GB/s` (PCIe Gen5) | **`450 GB/s` (`NVLink-C2C`，快 `7.0 倍`!)** |
| **`EP=72` MoE 单卡达标 `Goodput`** | `980 tok/s/GPU` | `2,380 tok/s/GPU` | `4,650 tok/s/GPU` | **`7,420 tok/s/GPU` ✓（H200 的 `3.12 倍`）** |
| **实测 P99 Decode `TPOT`** | `21.6 ms` | `13.6 ms` | `8.4 ms` | **`5.2 ms` ✓（单用户 `192 tok/s`！）** |
| **机柜供电与散热门槛** | `35~42 kW/柜`（风/液冷） | `40~45 kW/柜`（风/液冷） | `50~65 kW/柜`（液冷为主） | **`120~132 kW/柜`（强制全液冷 DLC + CDU）** |
| **每百万达标 Token 综合成本** | `$0.56 / 1M` | `$0.25 / 1M` | `$0.19 / 1M` | **`$0.13 / 1M` ✓（较 H200 降 `48%`、较 H100 降 `76.8%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **大参数 MoE（`>=405B/671B`）且具备 `125 kW/柜` 液冷机房条件时，优先采购 `GB200 / GB300 NVL72`**：将 `EP=64/72` 锁定在单柜铜缆 NVLink 5 域内。
- [ ] **充分利用 Grace CPU `450 GB/s/GPU` 的 `NVLink-C2C` 通道配置 `LMCache / Dynamo` 二级 KV 缓存池与 RL 权重快速卸载**。
- [ ] **同一 `NVL72` 机柜内部署作业时，通过 K8s `Kueue TAS` 绑定整柜或半柜拓扑域，避免跨机柜切分 `TP/EP`**。
