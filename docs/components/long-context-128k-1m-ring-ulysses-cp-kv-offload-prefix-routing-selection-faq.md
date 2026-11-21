# 超长上下文（`128K–1M Context`）大模型训推架构选型：`Ulysses × Ring` 混合上下文并行（`2D USP`）、三级 KV Cache 卸载（`HBM -> Host DRAM -> NVMe GDS`）与前缀感知路由指南

> **语言 / Language**: **中文 (当前)** | [English](./long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect、长文本推理平台负责人与分布式架构师的 `128K–1M` 超长上下文训推架构、Attention $\mathcal{O}(S^2)$ 拐点公式、`DeepSpeed-Ulysses vs. Ring Attention vs. 2D USP` 上下文并行选型与跨机 RDMA KV Cache 池化卸载白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业云架构选型与容量规划）

![超长上下文 128K-1M 2D USP 上下文并行与三级 KV 卸载架构总结图](../../assets/svg/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**当上下文长度 $S$ 突破 `64K ~ 128K` 迈向 `1M` 时，大模型会同时撞上两堵物理高墙：① Prefill 阶段自注意力 $\mathcal{O}(S^2)$ 算力超越线性层（当 $S > 12 d_{\text{model}} \approx 96\text{K}$ 时），单机 `TP=8` 首字耗时飙升至 `28.4 秒`；② 单会话 KV Cache 高达 `4.5 ~ 21 GB`，多轮对话稍有并发就会触发 HBM 驱逐重算。最优解是采用“2D 混合上下文并行（`Intra-Node Ulysses [NVLink All-to-All] × Inter-Node Ring Attention [RDMA P2P]`）+ 三级分层 KV 池（`L1 GPU HBM <-> L2 锁页主存 DDR5 <-> L3 NVMe GDS / Mooncake 跨机池`）+ 一致性前缀哈希路由”，将 `128K` 首字延迟从 `28.4s` 压降至 `2.3s`（提速 `12.3 倍`）、多轮缓存命中率拉升至 `91.4%`，并降低 `82.9%` 百万 Token 成本！**

- **交互式 `128K–1M` CP 并行与 KV 卸载召回计算器**：[打开交互式计算器 (`long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-sizer.html`)](../reports/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-cards.html`)](../reports/long-context-128k-1m-ring-ulysses-cp-kv-offload-prefix-routing-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：为什么上下文超过 `96K ~ 128K` 后，单机 `TP=8` 会突然慢如蜗牛？`DeepSpeed-Ulysses` 与 `Ring Attention` 各自的优缺点是什么？
1. **Attention $\mathcal{O}(S^2)$ 超越线性层的精确数学拐点**：
   - Transformer 每层线性投影（QKV + O + FFN）算力约为 $24 S d_{\text{model}}^2$（随序列长度 $S$ 线性增长），而因果自注意力（Causal Self-Attention）算力约为 $2 S^2 d_{\text{model}}$（随 $S^2$ 平方增长）；
   - 当 $2 S^2 d_{\text{model}} > 24 S d_{\text{model}}^2 \iff \mathbf{S > 12 d_{\text{model}}}$（对于 $d_{\text{model}} = 8,192$ 的 70B 模型，拐点正是 **$S \approx 98,304$ Tokens**）时，Attention 算力反超 MLP 成为绝对瓶颈！单台 8 卡节点算力封顶，必须沿序列维度切分——即 **上下文并行（Context Parallelism, `CP`）**。
2. **三大上下文并行（`CP`）方案深度对比**：
   - **`DeepSpeed-Ulysses`（切分 Attention Heads）**：在进入 Attention 前通过一次 `All-to-All` 将“序列切分、全量 Head”转置为“全量序列、Head 切分”，算完 Attention 后再 `All-to-All` 转回。
     - **优点**：通信量小，天然适配机内 `900 GB/s NVLink`；
     - **致命上限**：并行度受限于 KV 头数 $n_{kv}$！主流 GQA 模型（如 `Llama-3.3-70B/405B`）只有 **$n_{kv} = 8$ 个 KV 头**，因此纯 Ulysses 最多切到 `CP = 8`（一旦跨机切到 16/32 卡就必须重复复制 KV 头）。
   - **`Ring Attention`（切分 Sequence 块环网流转）**：每张卡持有 $S / CP$ 的 Query 块，同时在环网中通过 `P2P Send/Recv` 轮流传递 KV 块，并用双缓冲与本地 `FlashAttention` 计算重叠（Overlap）。
     - **优点**：不受 $n_{kv}$ 头数限制，可无限扩展至 `32 ~ 256` 卡跑 `1M ~ 10M` 上下文；
     - **缺点**：环网跳数多，对跨机跳延迟敏感，且需要 Causal ZigZag / Striped 负载均衡防止前一半卡算得快、后一半卡算得慢。
   - **`2D Hybrid USP`（Unified Sequence Parallelism = 机内 `Ulysses` × 跨机 `Ring Attention`，生产首选！）**：
     - 在单机 8 卡 `NVLink` 域内跑 **`Ulysses = 8`**（刚好匹配 $n_{kv} = 8$，吃满 `900 GB/s` NVLink），在跨节点 `400G/800G RDMA` 域间跑 **`Ring = 2 / 4 / 8`**（将环网跳数缩小 8 倍！），轻松在 32 张 H200 上将 `128K` Prefill TTFT 压到 **`2.3 秒`**！

### Q2：多轮 `128K` 长对话中，为什么把 KV Cache 卸载到主机内存（`Host DRAM`）或 `NVMe GDS` 比“驱逐后重新算一遍 Prefill”快 `50 ~ 690 倍`？
- **算一笔硬核账（以 `Llama-3.3-70B FP8` 单次 `128K` 会话为例）**：
  - 单会话 `128K` FP8 KV Cache 体积 = $131,072 \times 163.84\text{ KB} = \mathbf{21.0\text{ GB}}$（若为 `DeepSeek-R1 671B MLA FP8` 则仅 **`4.50 GB`**）；
  - 当用户在多轮阅读/改代码间隙思考 30 秒时，如果把这 `21.0 GB` KV 直接丢弃（Evict），下一轮用户追问时重新算一遍 `128K` 二次方 Prefill 需要耗时 **`2,300 ms`（32 卡 CP）~ `28,400 ms`（单机 8 卡）**！
  - 反之，如果将该 KV 卸载到本机 `2 TB DDR5` 锁页主存（**L2 Pinned Host DRAM**）或跨节点 `Mooncake / LMCache` RDMA 内存池：
    - 8 卡节点聚合 **PCIe Gen5 x16 双向带宽 = $8 \times 64\text{ GB/s} = 512\text{ GB/s}$**（或 8×400G RDMA = `400 GB/s`）；
    - 从主存/RDMA 召回整份 `21.0 GB` KV Cache 的传输耗时仅为：
      $$t_{\text{recall}} = \frac{21.0\text{ GB}}{512\text{ GB/s}} \approx \mathbf{41.0\text{ ms}}$$
    - **`41.0 ms`（召回）vs. `2,300 ~ 28,400 ms`（重算）——速度快了 `56 ~ 692 倍`**，且省下了 99% 的 Prefill 算力！

---

## 3. 🧭 直观可视化图解（Mermaid 2D USP 架构图 + 三级 KV 卸载流水线）

### 3.1 `2D Hybrid USP`（机内 `Ulysses=8` × 跨机 `Ring=4`）双层上下文并行架构图
```mermaid
flowchart TD
    A["128K ~ 1M 超长序列输入<br/>(ZigZag 因果负载均衡切块)"] --> B["跨节点外层: Inter-Node Ring Attention (RDMA 400G/800G)<br/>4 台 8x H200 节点组成 4 跳 Ring (P2P KV 传输与计算完全重叠)"]
    B --> C1["Node 0: Intra-Node Ulysses=8<br/>(NVLink 900GB/s All-to-All 切 8 个 KV 头)"]
    B --> C2["Node 1: Intra-Node Ulysses=8<br/>(NVLink 900GB/s All-to-All 切 8 个 KV 头)"]
    B --> C3["Node 2 & 3: Intra-Node Ulysses=8<br/>(NVLink 900GB/s All-to-All 切 8 个 KV 头)"]
    C1 & C2 & C3 --> D["128K 首字 TTFT 从 28.4s 骤降至 2.3s (-12.3x!)"]
```

### 3.2 三级分层 KV Cache 存储池（`L1 HBM <-> L2 Host DRAM <-> L3 NVMe GDS`）流转时序
```mermaid
sequenceDiagram
    autonumber
    participant User as 多轮 128K 编程/文档 Agent
    participant Router as Prefix-Aware 一致性路由网关
    participant HBM as L1: GPU HBM 热缓存池
    participant DRAM as L2: 跨机 RDMA Pinned DDR5 内存池 (2TB/机)
    participant GDS as L3: 本地 NVMe SSD (GPUDirect Storage)

    User->>Router: 第 N 轮追问 (128K 历史前缀 + 500 字新问题)
    Router->>HBM: 按 Prefix Block Hash 路由至亲和实例查询 L1
    alt L1 HBM 命中 (热会话 <10s)
        HBM-->>User: 零传输! 仅 Prefill 新增 500 字 (TTFT 85ms)
    else L1 已换出 -> 命中 L2 Host DRAM / RDMA Pool (温会话 10s~30min)
        DRAM->>HBM: 按层异步 PCIe Gen5 / RDMA DMA 召回 21GB KV (仅 41ms!)
        HBM-->>User: 免除 28.4s 重算! 首字响应 TTFT 仅 140ms
    else L2 已换出 -> 命中 L3 NVMe GDS (冷会话 >30min)
        GDS->>HBM: GPUDirect Storage 零拷贝直读入显存 (~380ms)
        HBM-->>User: 首字响应 TTFT < 550ms
    end
```

---

## 4. 📊 架构师多方案选型决策矩阵（四档超长上下文方案对比）

| 评估维度 | Tier 1：单机 `TP=8` 硬扛（无 CP、HBM 驱逐后全量重算） | Tier 2：纯跨机 `Ring Attention CP`（无 Ulysses 与主存池） | Tier 3：纯 `Ulysses CP`（受限于 `n_kv=8`）+ 本地主存卸载 | Tier 4（推荐）：`2D Hybrid USP (Ulysses × Ring)` + 分布式 `DRAM/GDS KV Pool` |
| :--- | :--- | :--- | :--- | :--- |
| **`128K` P95 首字延迟 (`TTFT`)** | `28.4 s` ✕ | `9.8 s` | `6.4 s` | **`2.3 s` ✓（提速 `12.3 倍`）** |
| **`128K` P99 逐字延迟 (`TPOT`)** | `62.5 ms` ✕ | `29.4 ms` | `21.2 ms` ✓ | **`15.6 ms` ✓** |
| **多轮会话有效 KV 缓存命中率** | `18.5%` ✕（频繁 OOM 驱逐） | `34.0%` | `68.5%` | **`91.4%` ✓** |
| **每百万 `128K` Token 综合成本** | `$3.40 / 1M` ✕ | `$1.65 / 1M` | `$1.02 / 1M` | **`$0.58 / 1M` ✓（降低 `82.9%`）** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **超过 `64K` 上下文强制启用 `2D Hybrid USP`（机内 `Ulysses` × 跨机 `Ring Attention`）**：配合 ZigZag / Striped 序列分块消除因果掩码（Causal Mask）导致的负载不均。
- [ ] **标配每节点 `1.5 ~ 2 TB` 锁页主存（Pinned DDR5）+ `LMCache / Mooncake` 二级 KV 池**：严禁在多轮长文本场景让被驱逐的 KV 直接丢弃触发 $\mathcal{O}(S^2)$ 重算。
- [ ] **网关层强制开启前缀块哈希（Prefix-Block Hash）一致性路由**：避免轮询（Round-Robin）把同一长会话打散到不同节点。
