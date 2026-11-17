# 企业级千亿参数 MoE 大模型（DeepSeek-V3/R1 671B、Qwen3-235B）生产部署与专家并行（TP vs. EP）架构选型指南：双机跨网纯 TP=16 vs. 单机 8xH200 机内 NVLink vs. 跨机静态 EP=16 vs. 宽专家并行（Wide EP + DeepEP 双微批重叠 + EPLB 动态冗余热点专家复制）架构

> **文档类型**：企业级 AI 基础设施与集群容量规划解决方案架构师（Cloud & AI Solution Architect）选型决策指南（FAQ）  
> **适用读者**：企业 AI 基础设施负责人、算力中心架构师、云计算选型与 IT 采购总监  
> **核心目标**：站在企业架构师选型与容量规划视角（不抠底层汇编与内核源码），彻底讲清楚为什么以 **DeepSeek-V3/R1（总参 671B / 激活 37B，256 个路由专家）**、**Qwen3-235B-A22B** 为代表的**稀疏混合专家模型（Sparse MoE）**，在生产部署时绝对不能照搬传统 70B 稠密模型的跨机张量并行（TP）套路——系统回答：**“为什么用两台 8xH100 (80GB) 跨机跑 `TP=16` 会慢得像蜗牛（单卡仅 148 tok/s）？”、“什么时候应该直接买单台 `8x H200 (1,128GB)` 靠机内 NVLink 一劳永逸解决问题，什么时候才需要上跨机 `EP=16/32` 宽专家并行？”、“什么是 MoE 生产环境最致命的‘热点专家木桶效应（Hot-Expert Straggler）’，怎么靠动态冗余专家复制（EPLB）把集群吞吐再提 59%？”**。  
> **标定声明**：本文所引用的单卡吞吐（`tok/s/GPU`）、P99 吐字间隔（TPOT）、热点专家负载不均衡比与支撑 `15,000 tok/s` 集群月度成本模型基于公开发布的 MoE 推理系统架构基准（DeepSeek-V3/R1 开源推理剖析报告、DeepEP 通信库、EPLB 专家负载均衡器及 SGLang / vLLM DP-Attention + EP 生产评测）构建的合成分析模型（`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`），详见 [`experiments/enterprise_moe_serving_expert_parallelism_tp_vs_ep_selection/results.json`](../../experiments/enterprise_moe_serving_expert_parallelism_tp_vs_ep_selection/results.json)。

---

## 一图流总结（Executive Visual Summary）

![企业级千亿 MoE 大模型并行架构选型总结](../../assets/svg/enterprise-moe-serving-expert-parallelism-tp-vs-ep-selection-faq-summary.svg)

> 🔗 **关联可交互报告与工具**：
> - 🃏 **[7-Card 4:5 视觉评测报告 (`enterprise-moe-serving-expert-parallelism-tp-vs-ep-selection-cards.html`)](../reports/enterprise-moe-serving-expert-parallelism-tp-vs-ep-selection-cards.html)**
> - 🧮 **[交互式 671B/235B MoE 机型与并行度选型计算器 (`enterprise-moe-serving-expert-parallelism-tp-vs-ep-sizer.html`)](../reports/enterprise-moe-serving-expert-parallelism-tp-vs-ep-sizer.html)**
> - 🌐 **[English Version (`enterprise-moe-serving-expert-parallelism-tp-vs-ep-selection-faq.en.md`)](./enterprise-moe-serving-expert-parallelism-tp-vs-ep-selection-faq.en.md)**

---

## ① 📌 核心选型结论速览（Executive Summary）

以 **DeepSeek-V3 / R1 (`671B` FP8 权重约 `671 GB`)** 为例：单台 `8x H100 / H800 (80GB × 8 = 640 GB)` 装不下 `671 GB` 的模型权重，必须跨两台 8xH100（共 16 卡，`1,280 GB`）或者买单台更大显存的 `8x H200 (141GB × 8 = 1,128 GB)`。

在支撑企业级 **`15,000 tok/s` 达标输出吞吐（要求 `P99 TPOT <= 30 ms`）** 的基准下，四档硬件与并行架构对比如下：

| 架构档位 | 核心硬件与并行策略 | 单卡有效吞吐 (tok/s/GPU) | 热点慢卡负载不均比 | P99 吐字间隔 (TPOT) | 支撑 15,000 tok/s 月成本 | 架构师拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **🟥 Tier 1：双机跨网纯张量并行 (`TP=16`)** | 2 台 8xH100 跨机硬切 `TP=16`（无 EP 优化） | `148 tok/s` ✕ | `1.00x` (全切碎) | **`82.0 ms` ✕ (跨机卡死)** | `$96,000 / 月` ✕ | **头号错误部署法**：每吐1个字跨网同步122次，专家切太碎 |
| **🟦 Tier 2：单机 `8x H200 (1,128GB)` 机内 NVLink** | **单台 8xH200 141GB (`TP=8 / EP=8`) 零跨机通信** | `606 tok/s` ✅ (提4.1x) | `1.35x` | `23.5 ms` ✅ | `$54,000 / 月` (-43.8%) ✅ | **`<= 24 卡` 中小集群绝对首选**：免去一切跨机 RDMA 调优！ |
| **🟨 Tier 3：跨机静态专家并行 (`EP=16` 无 EPLB)** | 2 台 8xH100 开启 `EP=16`，但专家固定死分配 | `465 tok/s` | **`3.40x` ✕ (木桶慢卡)** | `36.0 ms` (小幅超标) | `$64,000 / 月` | **受困于热点专家不均**：15张卡空等1张热门编程/财务专家卡 |
| **🟩 Tier 4：宽专家并行 (`EP=16/32`) + 双微批重叠 + `EPLB`** | **DP-Attention + DeepEP 双批重叠 + 动态冗余热点专家** | **`738 tok/s` 🏆 (提4.99x)** | **`1.08x` 🏆 (全群均衡)** | **`19.5 ms` 🏆** | **`$38,000 / 月` 🏆 (-60.4%)** | **大规模 H100/H800 算力中心终局标准**：榨干每一分算力 |

### 💡 架构师三句话选型铁律
1. **硬件选型捷径：如果企业总预算在 `1~3 台服务器（8~24 张卡）` 以内，请无脑采购单机 `8x NVIDIA H200 (141GB × 8 = 1,128 GB)`（Tier 2）！**：`671 GB` 的 FP8 权重放进单台 `1,128 GB` 的 H200 服务器后，还剩 **`457 GB` 显存全留给 KV Cache**！所有张量/专家通信 100% 走机内 `900 GB/s` 的 NVLink 总线，**完全不需要配置复杂的跨机 RoCEv2/InfiniBand RDMA 网络**，单卡吞吐直接从双机 H100 `TP=16` 的 `148 tok/s` 暴涨至 **`606 tok/s`（提升 `4.1 倍`）**！
2. **如果手头已经是 `80GB H100 / H800` 必须跨多机部署，严禁使用纯 `TP=16`（Tier 1）——必须切换为“注意力层数据并行（DP-Attention）+ 专家层专家并行（Expert Parallelism, EP）”**：DeepSeek-V3/R1 每层有 `256` 个细粒度专家，如果用 `TP=16` 把每个小专家再横切 16 份，单卡上的矩阵碎片小得连 GPU 计算单元都塞不满，且每层要跨机器同步两次；改为 **`EP=16`（每张卡完整托管 16 个专家，不切碎专家）+ `DP-Attention`（每张卡独立算不同请求的 Attention，不重复存储 KV Cache）**，是跨机跑通 MoE 的核心秘诀！
3. **跨机跑 EP 必须同时打开“双微批通信计算重叠（Dual-Microbatch Overlap / DeepEP）”与“动态冗余热点专家负载均衡（EPLB）”（Tier 4）**：真实企业流量会集中在某几个热门专家上（如专门写 Python 或算财务的专家被调用频率高出 `3.4 倍`），导致那 1 张卡忙死、其余 15 张卡干等（Tier 3）。开启 **`EPLB`（自动把最热的 Top 专家复制多份分散到空闲卡上）**，单卡吞吐再提 **`58.7%`（达 `738 tok/s/GPU`）**！

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（Architect Deep-Dive Q&A）

### Q1：很多运维同学部署完 70B 稠密模型后，用一模一样的命令（只把 `--tensor-parallel-size 8` 改成 `16`）跨两台 H100 部署 DeepSeek-671B（Tier 1），为什么吐字慢到没法看？
* **真相剖析（细粒度 MoE 架构与传统跨机 TP 的物理冲突）**：
  - 在传统 70B 稠密模型（Dense Model）中，每层只有 1 个巨大的 FFN 矩阵（中间维高达 `28,672`），切成 8 份或 16 份后每份矩阵依然很大。
  - 但在 **DeepSeek-V3/R1（细粒度 MoE）** 中，每层被拆成了 **`256 个` 微型路由专家**（每个专家的中间维只有 `2,048`）！
  - 如果你强行开 `TP=16`：每个 `2,048` 维的小专家被切成只有 **`128 维`** 的极薄切片！GPU Tensor Core 算 128 维的小矩阵连热身都不够，90% 的时间全花在等待跨网线的 `122 次/Token` AllReduce 同步上了（`P99 TPOT` 恶化至 `82 ms`）！
  - **换用专家并行 `EP=16` 后**：不切碎任何一个专家，每张 GPU 完整拥有 `256 / 16 = 16 个` 完整的 `2,048` 维专家，矩阵乘法效率瞬间拉满！

### Q2：什么是 MoE 推理中的“热点专家慢卡效应（Hot-Expert Straggler）”，为什么在企业私有化场景比公网更严重？
* **真相剖析（垂直业务流量高度偏斜导致的“一卡卡全群”）**：
  - 在公网闲聊场景中，有人问诗词、有人问做菜、有人问数学，256 个专家被激活的概率相对分散。
  - **但在企业内部生产场景中（比如全公司 5,000 名程序员都在用它写 Java 微服务，或者全银行都在用它审信贷合同）**，90% 的 Token 都会高频路由到那 **8~12 个专门负责 Java 语法或金融条款的“劳模专家（Hot Experts）”** 上！
  - 在静态 `EP=16`（Tier 3）中，存放“Java 专家”的 GPU-3 每轮要处理 **3,400 个 Token**，而存放“古诗词专家”的 GPU-11 每轮只有 **200 个 Token**。因为每层结束时所有 16 张卡必须等待 `All-to-All Combine` 同步，**15 张卡都在干等 GPU-3 算完**！
  - **Tier 4 的 `EPLB（Expert Parallelism Load Balancer）` 怎么解决？**
    - 引擎实时统计最近几分钟每个专家的激活热度；
    - 每张 GPU 额外预留 1~2 个冗余专家显存槽（仅占几百 MB），**自动把全群最热的“Java 专家”复制 3~4 份放到原本清闲的 GPU 上**，让路由器将发往该热门专家的 Token 均匀打散到多张卡上，将负载不均比从 `3.40x` 压平到 **`1.08x`**！

### Q3：什么是“双微批乒乓重叠（Two-Microbatch Ping-Pong Overlap）”，它是怎么把跨机 RDMA 网络延迟藏起来的？
* **真相剖析（让网卡传 Batch-A 的同时，GPU 计算单元在算 Batch-B！）**：
  - 在专家并行（EP）中，每层都需要先通过 RDMA 网络把 Token 发给对应的专家卡（`Dispatch All-to-All`），专家算完后再通过 RDMA 发回来（`Combine All-to-All`）。如果串行执行，GPU 有一半时间在等网卡收发包。
  - **Tier 4（SGLang / vLLM + DeepEP 双微批流水线）**：将当前正在处理的请求队列对半拆成 **Microbatch A** 和 **Microbatch B**：
    - 当 **Microbatch A** 正在走 RDMA 网卡做跨机 `All-to-All Dispatch` 通信时，GPU 的 Tensor Core 绝不闲着，立刻同时计算 **Microbatch B** 的 `MLA Attention / Shared Expert`！
    - 两个微批像打乒乓球一样交替掩盖对方的通信耗时，把跨机网络延迟 **100% 隐藏在计算背后**！

---

## ③ 🧭 直观可视化图解（Mermaid 决策树、TP vs EP 对比与双微批重叠图）

### 3.1 企业级千亿参数 MoE（DeepSeek-671B / Qwen3-235B）部署选型决策树

```mermaid
flowchart TD
    A[企业计划部署 DeepSeek-V3/R1 671B 或 Qwen3-235B MoE 大模型] --> B{企业采购预算与集群并发吞吐规模？}
    B -->|中小并发 (< 30 RPS) 或初期 PoC<br/>不想自建百卡机房| C[优先使用云厂商全托管 MaaS API<br/>或混合 PTU + PAYGO 溢出架构<br/>详见 Round 361]
    B -->|中高并发私有化自托管 (1~3 台服务器规模)| D{是否能直接采购 8x H200 (141GB) 或 B200 机型？}
    D -->|能采购 8x H200 (单机 1,128GB 显存)| E[🟦 Tier 2: 单机 8x H200 机内 NVLink (TP=8 / EP=8)<br/>单机完整装下 671B-FP8 + 457GB KV Cache<br/>零跨机网络依赖 · 开箱即获 4,850 tok/s/机 ✅]
    D -->|只能使用 80GB H100 / H800 跨多机部署<br/>或建设 >= 32 卡超大规模高吞吐算力中心| F[🟩 Tier 4: DP-Attention + 宽专家并行 (EP=16/32) + DeepEP 双批重叠 + EPLB<br/>消灭跨机 TP 碎矩阵 · 动态复制热点专家消除慢卡<br/>单卡吞吐 738 tok/s (提4.99x) · 成本省 60.4% 🏆]
```

### 3.2 为什么 MoE 必须用“DP-Attention + EP + EPLB”而非传统跨机 TP？架构对比图

```mermaid
flowchart LR
    subgraph BadTP["❌ Tier 1: 跨机纯张量并行 TP=16 (单卡仅 148 tok/s)"]
        T1["256 个微型专家每个都被横切 16 刀<br/>-> 变成 128 维极碎矩阵，算力空转 ✕<br/>-> 每层 2 次跨网 AllReduce 阻塞 (TPOT 82ms) ✕"]
    end

    subgraph GoodEP["🟩 Tier 4: DP-Attention + EP=16 + EPLB 热点复制 (单卡 738 tok/s 🏆)"]
        E1["🧠 Attention 层走数据并行 (DP=16)<br/>每张卡独立算不同请求 · KV Cache 零重复存储 ✅"]
        E2["⚡ DeepEP 双微批乒乓流水线<br/>Microbatch-A 跨网 RDMA 通信与 Microbatch-B 计算 100% 重叠"]
        E3["⚖️ MoE 专家层走专家并行 (EP=16) + EPLB<br/>每卡托管 16 个完整专家 + 自动多副本复制热门编程/财务专家 🏆"]
        E1 --> E2 --> E3
    end
```

### 3.3 Tier 4 双微批乒乓通信-计算重叠（Two-Microbatch Ping-Pong Overlap）时序图

```mermaid
sequenceDiagram
    autonumber
    participant Comp as GPU Tensor Core (计算引擎)
    participant RDMA as 400G RoCEv2/IB 网卡 (DeepEP 通信引擎)

    Note over Comp,RDMA: 将当前解码批次拆分为 Microbatch A 与 Microbatch B 交替掩盖延迟
    par 阶段 1：A 算注意力，B 传专家数据
        Comp->>Comp: 计算 Microbatch A 的 MLA Attention 与共享专家
    and
        RDMA->>RDMA: 跨机 All-to-All Dispatch 传输 Microbatch B 的激活值至目标专家卡
    end
    par 阶段 2：B 算路由专家，A 传专家数据
        Comp->>Comp: 计算 Microbatch B 的完整路由专家矩阵 (EPLB 负载均衡无慢卡)
    and
        RDMA->>RDMA: 跨机 All-to-All Dispatch 传输 Microbatch A 的激活值至目标专家卡
    end
    Note over Comp,RDMA: GPU 计算核心与 RDMA 网卡双双接近 100% 满载，P99 TPOT 压至 19.5ms 🏆
```

---

## ④ 📊 架构师多方案选型决策矩阵（Architect Decision Matrix）

| 评估维度 | 🟥 Tier 1：双机跨网纯 `TP=16` | 🟦 Tier 2：单机 `8x H200` 机内 NVLink | 🟨 Tier 3：跨机静态 `EP=16` (无 EPLB) | 🟩 Tier 4：宽 `EP` + `DeepEP` + `EPLB` |
| :--- | :--- | :--- | :--- | :--- |
| **推荐适用规模** | ❌ 严禁用于跨机 MoE | **`8~24 卡` 企业私有化首选** | ⚠️ 仅限均匀合成压测 | **`16~128+ 卡` 高并发算力中心首选** |
| **单卡实测有效吞吐** | `148 tok/s/GPU` ✕ | `606 tok/s/GPU` ✅（`4.1x`） | `465 tok/s/GPU` | **`738 tok/s/GPU` 🏆（比 Tier 1 高 `4.99x`）** |
| **热点专家慢卡负载比** | `1.00x`（但全被切碎） | `1.35x` | **`3.40x` ✕（1张热卡拖慢15张卡）** | **`1.08x` 🏆（EPLB 动态复制热门专家）** |
| **P99 吐字间隔 (TPOT)** | `82.0 ms` ✕（严重卡顿） | `23.5 ms` ✅ | `36.0 ms` | **`19.5 ms` 🏆（极致丝滑）** |
| **网络与运维复杂度** | 高（跨机延迟极敏感） | **极低（单机框内 `900 GB/s` NVLink）** | 高（需 RoCEv2/IB RDMA） | 高（需 200G/400G RDMA + EP 编排） |
| **支撑 15,000 tok/s 月成本** | `$96,000 / 月` ✕ | `$54,000 / 月` ✅（省 `43.8%`） | `$64,000 / 月` | **`$38,000 / 月` 🏆（比 Tier 1 省 `60.4%`）** |

---

## ⑤ 🧪 容量/成本拐点算例与 7-Card 报告（Quantitative Sizer & 7-Card Deck）

### 5.1 基准算例假设（671B-FP8 MoE 模型 · 企业目标产能 `15,000 tok/s` · SLA `P99 TPOT <= 30 ms`）
- **架构师采购拍板精算**：
  1. **如果你只需要 `4,000 ~ 10,000 tok/s` 的吞吐（大多数大型企业的内部规模）**：直接采购 **1~2 台 `8x H200 (141GB)` 服务器（Tier 2）**。虽然 H200 单机租金比 H100 略贵 15%，但因为它把整套 671B 模型装进了单机 NVLink 域内，免去了购买昂贵的无损 InfiniBand/RoCE 交换机与雇佣 RDMA 网络专家的隐性成本，综合性价比无敌！
  2. **如果你要利用存量 `80GB H100/H800` 或建设 `15,000+ tok/s` 集团算力底座**：采用 **Tier 4（`DP-Attention + EP + DeepEP + EPLB`）**，单卡产出从 `148 tok/s` 暴涨至 `738 tok/s`，月成本从 `$96,000` 压至 **`$38,000/月`（立省 `60.4%`）**！

```chart
{
  "data": {
    "values": [
      { "Tier": "1. 双机跨网纯TP=16", "单卡吞吐(十tok/s)": 14.8, "P99吐字间隔TPOT(ms)": 82.0, "15K吞吐月成本($K)": 96.0 },
      { "Tier": "2. 单机8xH200机内NVLink", "单卡吞吐(十tok/s)": 60.6, "P99吐字间隔TPOT(ms)": 23.5, "15K吞吐月成本($K)": 54.0 },
      { "Tier": "3. 跨机静态EP=16(无EPLB)", "单卡吞吐(十tok/s)": 46.5, "P99吐字间隔TPOT(ms)": 36.0, "15K吞吐月成本($K)": 64.0 },
      { "Tier": "4. 宽EP+双批重叠+EPLB", "单卡吞吐(十tok/s)": 73.8, "P99吐字间隔TPOT(ms)": 19.5, "15K吞吐月成本($K)": 38.0 }
    ]
  },
  "semantic_types": {
    "Tier": "Category",
    "单卡吞吐(十tok/s)": "Quantity",
    "P99吐字间隔TPOT(ms)": "Quantity",
    "15K吞吐月成本($K)": "Amount"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "宽专家并行 + DeepEP 双微批重叠 + EPLB (Tier 4) 将 671B 单卡吞吐提速 4.99x 且月省 60.4%",
    "subtitle": "671B-FP8 MoE 模型下的单卡吞吐 (×10 tok/s)、P99 TPOT (ms) 与 15K tok/s 达标月成本 ($K)",
    "encodings": {
      "x": "Tier",
      "y": ["单卡吞吐(十tok/s)", "P99吐字间隔TPOT(ms)", "15K吞吐月成本($K)"]
    }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地 Checklist（Production Readiness Checklist）

在企业规划与上线千亿参数 MoE 大模型（DeepSeek-V3/R1 671B、Qwen3-235B）前，架构师必须逐项核对以下 8 条铁律：

- [ ] **1. 新采购单机柜私有化服务器时，优先选配 `8x H200 (141GB)` 或 `B200` 取代双机 `80GB H100`**：让 `671 GB` 的 FP8 权重 100% 装进单机 NVLink 域内，从物理上消灭跨机通信瓶颈。
- [ ] **2. 跨多台 80GB 节点部署细粒度 MoE 时，严禁无脑使用跨机纯张量并行（`TP=16`）**：必须开启专家并行（`--enable-expert-parallel` / `EP=16`），避免将微型专家切碎成无法跑满算力的极薄矩阵。
- [ ] **3. 开启注意力层数据并行（`--enable-dp-attention` / `DP-Attention`）**：让不同 GPU 独立处理不同请求的 MLA/GQA 注意力层，避免在 16 张卡上重复复制相同的 KV Cache。
- [ ] **4. 生产环境强制开启 `EPLB`（动态冗余热点专家负载均衡）**：针对企业垂直业务流量偏斜，自动在每张卡预留冗余槽位复制高频激活的“热门专家”，消除 `All-to-All` 木桶慢卡效应。
- [ ] **5. 启用双微批通信-计算重叠（`Two-Batch Overlap / DeepEP`）**：将跨节点 RDMA `Dispatch/Combine` 通信耗时隐藏在另一微批的 Attention 与共享专家计算背后。
- [ ] **6. MoE 路由门控网络（Router Gate）与归一化层必须保持 `BF16 / FP32` 高精度**：即使 256 个专家权重使用 `FP8` 量化，负责给 Token 分配专家的路由器绝不能低比特量化（防止选错专家！）。
- [ ] **7. 结合 Round 385 规范开启 DeepSeek 原生 `MTP`（Multi-Token Prediction）投机解码**：将 MoE 模型的单请求流式生成速度再提升 `1.8x~2.4x`。
- [ ] **8. 跨机 EP 集群的网络交换机必须开启 RoCEv2 PFC/ECN 无损流控或使用 InfiniBand**：防止高频并发 `All-to-All` 流量引发交换机微突发丢包重传（Incast Congestion）。
