# MoE（混合专家模型）训练与推理全栈架构选型：`DeepEP` 双模式通信（Normal vs. `IBGDA`）、`DualPipe / 双批次乒乓重叠`、无辅助损失路由与 `EPLB` 动态专家重排指南

> **语言 / Language**: **中文 (当前)** | [English](./moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与大规模分布式训推负责人的 MoE 系统工程四支柱（`DeepEP Normal/IBGDA`、`DualPipe / Two-Batch Ping-Pong Overlap`、`Auxiliary-Loss-Free Routing` 与 `Hierarchical EPLB`）架构选型与通信带宽手算白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业 MoE 集群架构规划与选型）

![MoE 全栈架构选型 DeepEP IBGDA、双批次重叠与 EPLB 总结图](../../assets/svg/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**MoE 模型（如 `DeepSeek-V3/R1 671B`、`Qwen3-235B-A22B`）之所以能用 `37B` 激活参数打出 `671B` 的智能水平，代价是把计算压力转移给了每层 2 次跨卡 `All-to-All`（`Dispatch` 分发 + `Combine` 归并）通信与专家冷热不均木桶效应！如果直接用通用 `NCCL All-to-All` 串行跑，`54.8%` 的时间全在空等网络！企业生产落地大规模 MoE 必须凑齐四大支柱：① `DeepEP` 双模式通信内核（Prefill 用 `NVLink->RDMA` 分层转发吃满带宽；Decode 用 `IBGDA` GPU 线程直敲网卡门铃将单次延迟从 `820us` 压至 `168us`）；② `DualPipe / 双批次乒乓重叠（Two-Batch Ping-Pong Overlap）`（用微批次 B 的计算完美掩盖微批次 A 的 RDMA 传输，将暴露通信停顿压至 `1.8%`）；③ 训练侧 `无辅助损失动态偏置路由（Auxiliary-Loss-Free Bias）`；④ 推理侧 `EPLB 分层冗余热专家复制`（将最忙卡负载偏斜从 `3.85x` 压平至 `1.09x`），使单卡达标 Goodput 提升 `4.58 倍`（`2,380 tok/s/GPU`）、成本降低 `78.8%`！**

- **交互式 MoE `DeepEP` 通信与 `EPLB` 负载均衡计算器**：[打开交互式计算器 (`moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-sizer.html`)](../reports/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-cards.html`)](../reports/moe-architecture-deepep-ibgda-dualpipe-eplb-load-balancing-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：每跑一层 MoE，`Dispatch + Combine` 到底要在网络上搬运多少数据？为什么 Prefill 和 Decode 必须用两套完全不同的 `DeepEP` 内核？
1. **单层单 Token 的 MoE `All-to-All` 通信量精确手算**（以 `DeepSeek-V3/R1` 为例：隐藏维度 $d = 7,168$，每个 Token 激活 $K = 8$ 个路由专家，跨 $EP = 64$ 张卡）：
   - **Dispatch（分发，采用 `FP8 = 1 Byte`）**：每个 Token 需向 8 个目标专家发送 $8 \times 7,168 \times 1\text{ B} = 57.34\text{ KB}$；
   - **Combine（归并，采用 `BF16 = 2 Bytes` 保持累加精度）**：8 个专家算完后发回 $8 \times 7,168 \times 2\text{ B} = 114.69\text{ KB}$；
   - **单层单 Token 跨卡总通信量**：$(57.34 + 114.69) \times \frac{EP-1}{EP} \approx \mathbf{169.3\text{ KB / token / layer}}$（58 层 MoE 合计每 Token 需跨卡传输 **`9.82 MB`** 数据！）。
2. **为什么 `DeepEP` 拆分为 `Normal（高吞吐）` 与 `Low-Latency（IBGDA 低延迟）` 两种模式？**：
   - **Prefill / 训练阶段（大包带宽受限 $\to$ 用 `DeepEP Normal`）**：一次分发几千个 Token，先把去往同一台远端节点的最多 4 个专家的 Token 在**机内 `900 GB/s NVLink` 合并去重**，再通过配对 `400G RDMA` 网卡发出（仅占 `20~24 个 SM`，跑满 `46 GB/s` 线速）；
   - **Decode 阶段（微包极低延迟受限 $\to$ 用 `DeepEP Low-Latency IBGDA`）**：Decode 每次每卡只有几十个 Token，传统通过 CPU 代理线程（Proxy Thread）下发 RDMA WQE 的唤醒延迟高达 `820 us`！开启 **`IBGDA`（InfiniBand GPUDirect Async）** 后，由 **GPU SM 线程直接绕过 CPU 写入网卡 PCIe BAR 门铃寄存器（Doorbell）**，单次跨机 `All-to-All` 延迟骤降至 **`168 us`**！

### Q2：什么是 `DualPipe`（训练）与 `Two-Batch Ping-Pong Overlap`（推理双批次乒乓重叠）？如何把 `54.8%` 的通信停顿压缩到 `1.8%`？
- **核心原理**：如果整批请求（Batch）一起做 `Attention -> Dispatch (等网络) -> MoE FFN -> Combine (等网络)`，在等待 RDMA 传包的百微秒里，GPU 的 TensorCore 完全处于饥饿空转状态！
- **双微批次乒乓交错（Micro-Batch A <-> Micro-Batch B）**：
  - 将当前活跃请求切分为两个微批次 **`Batch A`** 与 **`Batch B`**；
  - 当 **`Batch A`** 正在通过网卡执行跨机 `DeepEP Dispatch / Combine` 时，GPU 的 TensorCore 立即切换去全速计算 **`Batch B`** 的 `MLA Attention` 或 `Shared/Routed Expert GEMM`！计算与通信在时间轴上呈完美拉链式重叠，使暴露在关键路径上的未掩盖通信停顿从 `54.8%` 暴跌至 **`1.8%`**！

### Q3：当线上突然涌入大量同领域请求（如全员写 Python 代码），某几个“代码专家”所在的 GPU 被打爆成为慢节点（Straggler），怎么用 `EPLB` 解决？
- **训练阶段（`Auxiliary-Loss-Free` 无辅助损失负载均衡）**：为每个专家 $i$ 引入一个动态偏置项 $b_i$，仅在选 Top-K 时按 $\text{TopK}(s_{i,t} + b_i)$ 路由（若某专家过载则小幅下调 $b_i$，欠载则上调 $b_i$），而做门控加权时仍乘原始亲和度 $s_{i,t}$——既实现训练均衡，又**完全不污染主损失函数梯度**！
- **推理阶段（`Hierarchical EPLB` 分层专家并行负载均衡器）**：
  - 在 64 张 GPU 上部署 256 个路由专家时，本来每张卡只需放 $256 / 64 = 4$ 个专家；
  - `EPLB` 让每张卡额外多留 **1 个冗余槽位（即每卡存 5 个专家，全集群共 320 个槽位，多出 64 个槽位专门复制 Top-5% 最繁忙的热门专家）**！每隔几分钟根据滑动窗口路由统计热更新一次冗余专家副本，将最忙卡与平均卡的负载比（Skew Ratio）从 **`3.85x` 直接熨平到 `1.09x`**！

---

## 3. 🧭 直观可视化图解（Mermaid 双批次乒乓重叠时序 + EPLB 冗余复制）

### 3.1 `Two-Batch Ping-Pong Overlap` 计算与通信拉链式重叠时序图
```mermaid
sequenceDiagram
    autonumber
    participant TC as GPU TensorCores (计算流: MLA / Expert GEMM)
    participant RDMA as DeepEP IBGDA / RDMA 引擎 (通信流: Dispatch / Combine)

    Note over TC,RDMA: 乒乓重叠阶段 1: Batch A 算完 Attention 发包，Batch B 接管 TensorCore!
    TC->>RDMA: 触发 Batch A 的 DeepEP Dispatch (跨机分发 Token)
    par 通信流与计算流 100% 并行执行
        RDMA->>RDMA: 后台传输 Batch A Dispatch (168 us)
    and
        TC->>TC: 全速计算 Batch B 的 MLA Attention + 共享专家!
    end
    Note over TC,RDMA: 乒乓重叠阶段 2: Batch A 到达专家开始算 FFN，Batch B 开始发包!
    par 通信流与计算流继续交错
        TC->>TC: 全速计算 Batch A 的 Routed Experts FFN GEMM!
    and
        RDMA->>RDMA: 后台传输 Batch B 的 DeepEP Dispatch (168 us)
    end
```

---

## 4. 📊 架构师多方案选型决策矩阵（64x H200 `EP=64` 跑 `DeepSeek-R1 671B` 四档对比）

| 评估维度 | Tier 1：通用 `NCCL All-to-All`（无重叠、静态专家分布） | Tier 2：仅启用 `DeepEP` 内核（无乒乓重叠与 `EPLB`） | Tier 3：`DeepEP` + 双批次乒乓重叠（无动态 `EPLB`） | Tier 4（推荐）：`DeepEP (Normal+IBGDA)` + 双批次重叠 + `EPLB` 冗余热专家 |
| :--- | :--- | :--- | :--- | :--- |
| **未掩盖的暴露通信停顿占比** | `54.8%` ✕ | `24.6%` | `6.8%` | **`1.8%` ✓** |
| **Decode 单次 `All-to-All` 延迟** | `820 us` ✕（CPU 代理瓶颈） | `195 us` | `182 us` | **`168 us` ✓（纯 GPU `IBGDA`）** |
| **热门专家木桶偏斜比 (`Max/Mean`)** | `3.85x` ✕（单卡排队堵死全网） | `3.40x` ✕ | `2.95x` ✕ | **`1.09x` ✓（冗余专家分流）** |
| **单卡达标有效吞吐 (`Goodput`)** | `520 tok/s/GPU` | `1,180 tok/s/GPU` | `1,760 tok/s/GPU` | **`2,380 tok/s/GPU` ✓（Tier 1 的 `4.58 倍`）** |
| **每百万 Token 综合成本** | `$1.18 / 1M` | `$0.52 / 1M` | `$0.35 / 1M` | **`$0.25 / 1M` ✓（降低 `78.8%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **Decode 池强制开启 `NVSHMEM / DeepEP IBGDA`（`NVSHMEM_IB_ENABLE_IBGDA=1`）**：消除 CPU Proxy 线程唤醒延迟。
- [ ] **启用双微批次乒乓重叠（`--enable-two-batch-overlap` / `TBO`）**：确保 `All-to-All` 通信被另一半批次的 `MLA / FFN` 计算隐藏。
- [ ] **开启 `EPLB` 冗余专家槽位（如 256 专家配 288~320 槽位）**：防止业务流量分布突变（Domain Shift）触发单卡热专家雪崩。
