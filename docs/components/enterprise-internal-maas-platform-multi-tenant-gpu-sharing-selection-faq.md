# 多团队共享企业级模型服务中台（`Internal MaaS Platform`）与 GPU 多租户切分架构选型指南：部门烟囱独占集群 vs. 容器时间片轮转（`Time-Slicing / MPS` 互相挤崩反模式）vs. 纯静态 `NVIDIA MIG` 切卡 vs. “`API/Token 层共享底座` + 硬件级 `MIG` 跑小模型 + `Kueue` 队列弹性借用”三层 MaaS 中台架构

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-internal-maas-platform-multi-tenant-gpu-sharing-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、企业 AI 平台工程负责人（Platform Engineering Lead）与算力底座架构师的“如何把 25 个部门烟囱整合为一套高利用率内部 MaaS 平台、何时用 API 共享 vs. 何时用 MIG 切卡 vs. 何时用 Kueue 队列调度”实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection/results.json`](../../experiments/enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection/results.json)。

![多团队共享企业级模型服务中台（Internal MaaS Platform）与 GPU 多租户切分选型总结图](../../assets/svg/enterprise-internal-maas-platform-multi-tenant-gpu-sharing-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

当一家中大型企业的 **25 个内部研发与业务团队** 同时推进 AI 应用落地时，如果平台团队直接“按部门发 GPU 机器”，会迅速演变成一场算力灾难：
1. **乱象一：部门烟囱式独占 GPU 节点（`1 个团队 = 1 套独立 GPU 节点池`，Tier 1，全局 GPU 利用率仅 `22.0%`，80 张 H100 月烧 `$310,000`）**：25 个团队里其实有 **85% 的需求都在调同一批通用基础模型**（如 `Llama-3.3-70B / Qwen2.5-72B`、`Coder-32B`、`BGE-M3` 向量化与 Reranker）。如果给每个部门发几张专属 H100，相当于把同一份 72GB 的模型权重在显存里**重复加载了 25 遍**！每个部门平时的流量只有 0.5 QPS，显卡 78% 的时间在空转，而新项目申请一套环境要走 **14 天** 的审批与装机流程！
2. **乱象二：在 Kubernetes 上盲目开启“软件级 GPU 时间片轮转（`Time-Slicing`）或 `CUDA MPS`”让多团队混跑（Tier 2，邻里干扰与 `CUDA OOM` 相互挤崩率高达 `31.5%`）**：为了省卡，运维把 A 团队的 OCR 模型和 B 团队的 7B 模型用 Time-Slicing 塞进同一张卡。因为软件时间片**根本不隔离显存地址空间与故障域**，只要 A 团队发来一张超大分辨率图片导致显存突增，同卡上 B 团队和 C 团队的生产 Pod 瞬间全部陪葬报 `CUDA out of memory` 崩溃！
3. **乱象三：试图仅靠静态硬件 `NVIDIA MIG` 切分解决所有问题（Tier 3）**：MIG 切出的 `1g.10gb / 2g.20gb` 小切片根本装不下 `70B` 主力大模型，且静态划给各部门的切片无法在空闲时互相借用。

在 **支撑 25 个内部团队（85% 共享基础模型调用 + 15% 自研小模型与微调评测任务）** 的企业基准下，四档内部 MaaS 平台架构的量化对决如下：

| 方案档位 | 核心多租户共享与隔离组合 | 全集群平均 GPU 利用率 | 邻里干扰与 `CUDA OOM` 误伤率 | 新团队接入交付周期 | 所需 H100 总规模 (`25个团队`) | 月度平台总成本 (`TCO`) | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：部门烟囱独占节点 (`Siloed GPU Nodes`)** | 25 个团队各自独占服务器、重复部署相同 70B 模型 | `22.0%` ✕ *(25份重复权重占满显存)* | `0.0%` | `14.0 天` ✕ *(走流程等分卡)* | `80 张 H100` ✕ | `$310,000 / 月` ✕ | ⛔ 企业算力永远不够用的罪魁祸首 |
| **🟨 Tier 2：软件时间片混跑 (`GPU Time-Slicing / MPS`)** | 多租户容器共享同一显存空间与时间片 | `68.0%` | `31.5%` ✕ *(一处显存泄漏全卡陪葬)* | `3.0 天` | `40 张 H100` | `$178,000 / 月` ✕ *(含故障恢复)* | ⛔ 严禁用于多团队生产服务，故障域完全不隔离 |
| **🟦 Tier 3：纯静态硬件 `NVIDIA MIG` 切分 (`MIG Only`)** | 所有显卡静态切成固定小块分给各团队 | `54.0%` | `0.0%` *(硅片级硬件隔离)* | `2.0 天` | `48 张 H100` | `$164,000 / 月` | ⚠️ 适合小模型，但装不下 70B 且无法跨部门借卡 |
| **🟩 Tier 4：三层企业级 `MaaS` 中台 (`API 虚拟化共享池` + `MIG` 小模型 + `Kueue` 弹性借用)** | **L1 网关虚拟 Key 共享 `70B/向量` 大池 (`85%` 流量) + L2 硬件 `MIG` 隔离专有小模型 + L3 `Kueue` 队列跨部门闲置借卡** | **`86.5%` 🏆** *(提升3.9倍)* | **`0.0%` 🏆** *(API限流+MIG硬隔离)* | **`30 分钟 (0.02天)` 🏆** *(自助领Key)* | **`24 张 H100` 🏆** *(砍掉70%显卡)* | **`$82,000 / 月` 🏆** *(-73.5%)* | 🌟 **企业级多团队大模型服务中台唯一标准范式** |

> 💡 **架构师一句话选型铁律**：**“建设企业内部多团队共享大模型中台（Internal MaaS Platform），核心秘诀是‘在最高的抽象层做共享（Share at the Highest Possible Layer）’：① 85% 的通用大模型、Embedding 与 Reranker 需求，绝不给业务团队发卡，而是做成**统一高吞吐 API 共享池**，通过自助门户发放带配额与分账标签的 `Virtual API Key`（30 分钟自助接入，配合 `Chunked Prefill + Multi-LoRA`）；② 业务团队自研的专属小模型（`<= 8B` / OCR / 语音），用硬件级 **`NVIDIA MIG (1g.10gb / 2g.20gb)`** 做硅片级显存与算力硬隔离（0% OOM 误伤）；③ 微调与批量评测任务，用 **`Kubernetes Kueue`** 实现跨部门闲置 GPU 自动借用与 30 秒高优抢占归还！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大重复建群、切卡崩溃与影子 AI 泛滥痛点）

### Q1：为什么同样是让 25 个团队共用 GPU，“在 API / Token 层共享同一个 `vLLM` 池”比“把 GPU 切成小块发给 25 个团队自己部署”省下 70% 的显卡？怎么防止某个团队发超长 Prompt 卡死其他团队？
- **大白话与第一性原理**：
  1. **消灭“25 份重复模型权重税”**：一个 `70B FP8` 模型静态权重就要占掉 `72 GB` 显存。如果给 25 个部门各分 2 张卡各自部署，光静态权重就白白吃掉了 $25 \times 72\text{GB} = \mathbf{1,800\text{ GB}}$ 的昂贵 HBM3 显存！而如果由平台中台统一部署一套共享 `70B vLLM / SGLang` 服务池（哪怕个别部门有微调需求，也只挂载 160MB 的动态 `LoRA` 适配器），25 个团队的请求在同一个 Continuous Batch 里天然削峰填谷，仅需 **24 张 H100** 就能撑起原先 80 张卡的业务量（集群利用率从 `22%` 跃升至 **`86.5%`**）！
  2. **防邻里干扰双保险（`GCRA` 网关限流 + `Chunked Prefill` 分块预填充）**：
     - 在 AI 网关层按每个团队的 `Virtual API Key` 执行 **GCRA 令牌桶配额**（保底速率 + 允许突发借用空闲容量）；
     - 在 `vLLM / SGLang` 引擎层强制开启 **`--enable-chunked-prefill`（分块预填充）**：即便 A 团队突然发来一个 128K Token 的超长文档，引擎也会将其切成每次 `2,048 Token` 的微块与 B、C 团队的短对话交替执行，彻底消灭长请求阻塞短请求的队头阻塞（**邻里延迟干扰率 `0.0%`**）！

### Q2：对于确实不能用公共大模型、必须部署自己容器的专属小模型（如自研票据 OCR、风控 7B 小模型），为什么绝对不能用 `GPU Time-Slicing`，而必须用硬件 `NVIDIA MIG`？
- **软件时间片（Time-Slicing / MPS）vs. 硅片级物理隔离（Hardware `MIG`）的本质区别**：
  - **`Time-Slicing` 的致命缺陷**：多个容器共用同一个 GPU 显存控制器、同一块 L2 缓存和同一个虚拟地址空间上下文切换。一旦某个容器发生显存越界、死循环或 `CUDA OOM`，整张卡上的所有租户一起崩溃（误伤率 `31.5%`）。
  - **`NVIDIA MIG`（Multi-Instance GPU）的硬件级隔离**：把一张 `80GB H100` 在物理芯片层切成最多 7 个独立的 `1g.10gb` 或 4 个 `2g.20gb` 实例。每个 MIG 实例拥有**物理独立的显存条通道、独立的 L2 Cache 分区和独立的计算单元（SM）**——哪怕 A 团队在自己的 `1g.10gb` MIG 切片里把显存撑爆 100 次，隔壁 `2g.20gb` 切片里的生产服务连 1 毫秒抖动都不会有（**`0.0%` 故障传导**）！

### Q3：如果各事业部自己出钱买了专属 GPU 配额（不愿充公），怎么让他们在平时空闲时把卡借给别的部门跑微调/评测，一旦自己要用时又能在 30 秒内立刻收回？
- **`Kubernetes Kueue` 联盟配额借用与优先级抢占（Cohort Fair-Sharing & Preemption）**：
  - 为每个出资部门创建一个 `ClusterQueue`（保底配额 `nominalQuota`），并将所有部门归入同一个共享联盟 `Cohort: enterprise-ai-pool`。
  - 当零售部门的 8 张卡在下午闲置时，风控部门的 `P2` 微调与批量评估作业可以自动通过 `Kueue` **跨队列借用（Borrowing）** 这 8 张卡跑满算力；
  - 一旦零售部门提交了自己的高优任务（`P0/P1`），`Kueue` 立即触发 **30 秒优雅抢占（Graceful Preemption with Checkpoint）**，自动保存风控作业的 Checkpoint 并瞬间将 GPU 物归原主！

---

## ③ 🧭 直观可视化图解（决策树 + 三层 MaaS 中台参考架构图 + 自助接入与跨部门借卡时序图）

### 3.1 企业内部多团队 MaaS 平台与 GPU 共享隔离选型决策树

```mermaid
flowchart TD
    A["企业内部多团队 AI 算力与模型需求<br/>(25 个研发/业务团队 + 严控算力成本)"] --> B{"业务团队具体需求类型是什么?"}
    B -- "1. 调用通用大模型 / 代码模型 / 向量化 / Rerank (占 85%)" --> C["🌐 第一层：API / Token 层共享中台 (Shared MaaS API Pool)<br/>自助门户 30 分钟发放 Virtual API Key + 动态 Multi-LoRA + Chunked Prefill<br/>(严禁给每个部门发独立显卡重复部署 70B!)"]
    B -- "2. 部署团队自研专属小模型 (<=8B / OCR / 语音，占 10%)" --> D["🧩 第二层：硅片级硬件 NVIDIA MIG 物理切分池<br/>按需分配 1g.10gb / 2g.20gb / 3g.40gb 独立硬件切片<br/>(0.0% 显存 OOM 误伤，严禁用软件 Time-Slicing 跑生产!)"]
    B -- "3. 突发性模型微调 (SFT) / 批量离线评测任务 (占 5%)" --> E["⚖️ 第三层：Kubernetes Kueue 联盟弹性借卡池 🏆<br/>空闲时跨部门借用闲置配额跑满 86.5% 利用率；<br/>原主高优任务到达时 30s 自动保存断点并归还显卡!"]
    style C fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style D fill:#1e3a8a,stroke:#3b82f6,stroke-width:2px,color:#f8fafc
    style E fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
```

### 3.2 Tier 4 三层企业级内部 `MaaS` 服务中台（`API Pool + NVIDIA MIG + Kueue`）架构图

```mermaid
flowchart LR
    subgraph Portal["1. 自助式开发者门户与 AI 网关 (30m 接入)"]
        Teams["25 个内部业务团队"] --> DevPortal["🔑 自助式 MaaS 门户 (Backstage / APIM / LiteLLM)<br/>一键申领 Virtual API Key (自动绑定部门成本中心、<br/>GCRA 令牌桶配额、安全护栏与 OTel 追踪)"]
    end

    subgraph Layers["2. 三层差异化 GPU 共享与硬件隔离底座 (仅需 24x H100)"]
        DevPortal -->|85% 通用/LoRA 推理| L1["🚀 第一层：共享大模型推理池 (vLLM / SGLang)<br/>70B + Coder-32B + BGE-M3 + Multi-LoRA<br/>开启 Chunked Prefill 消除长短请求相互阻塞"]
        DevPortal -->|10% 自研专属小模型容器| L2["🛡️ 第二层：NVIDIA MIG 硬件切分池<br/>H100 切分为 1g.10gb / 2g.20gb / 3g.40gb<br/>独立显存控制器与 SM (0.0% OOM 邻里干扰)"]
        DevPortal -->|5% 微调/批量评测作业| L3["🔄 第三层：Kubernetes Kueue 弹性借用池<br/>跨部门 Cohort 闲置算力借用 + 30s 优先级抢占"]
    end
```

### 3.3 从“30 分钟自助上线新项目”到“`Kueue` 跨部门借卡与秒级抢占归还”全链路时序图

```mermaid
sequenceDiagram
    autonumber
    participant TeamA as 零售团队 (出资拥有 8 卡配额)
    participant TeamB as 风控团队 (新立项团队)
    participant Portal as 内部 MaaS 自助门户与 AI 网关
    participant Pool as 第一层共享 70B MaaS 池 (Chunked Prefill)
    participant Kueue as 第三层 K8s Kueue 联盟调度器

    TeamB->>Portal: 登录内部自助门户申请调用 `Llama-3.3-70B` 与 `BGE-M3`
    Portal-->>TeamB: 30 分钟内自动签发带 `$500/月` 预算上限与 `10,000 TPM` 配额的 Virtual API Key（无需买卡等 14 天！）
    TeamB->>Pool: 直接调用共享 70B 池（vLLM Chunked Prefill 保障与零售团队 0.0% 延迟互扰）
    Note over TeamA,Kueue: 下午 14:00 零售团队的 4 张微调/批处理 GPU 处于空闲状态
    TeamB->>Kueue: 提交紧急 LoRA 微调作业（标记为 `P2 Borrowed`）
    Kueue->>Kueue: 自动将零售团队闲置的 4 张卡借给风控团队运行（集群 GPU 利用率拉升至 86.5%！）
    TeamA->>Kueue: 16:00 零售团队提交本部门 `P0` 高优生产任务
    Kueue->>TeamB: 向风控借用 Pod 发送 SIGTERM，30 秒内将微调进度保存至 Checkpoint 并立即把 4 张卡归还给零售团队！
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：部门烟囱独占集群 | 🟨 Tier 2：软件时间片混跑 (`Time-Slicing`) | 🟦 Tier 3：纯静态硬件 `MIG` 切卡 | 🟩 Tier 4：三层企业级 `MaaS` 中台 (`API + MIG + Kueue`) |
|---|---|---|---|---|
| **全集群平均 GPU 利用率** | `22.0%` ✕（大量重复权重空转） | `68.0%` | `54.0%`（碎片化无法互借） | 🏆 **`86.5%`（API 池集中复用 + Kueue 跨部门填谷）** |
| **邻里干扰与 `CUDA OOM` 误伤** | `0.0%` | `31.5%` ✕（一崩俱崩） | `0.0%` | 🏆 **`0.0%`（Chunked Prefill + 硅片级 MIG 硬隔离）** |
| **新项目上线交付周期** | `14.0 天` ✕（采购/审批/装机） | `3.0 天` | `2.0 天` | 🏆 **`30 分钟 (0.02天)`（自助门户秒级领 Virtual Key）** |
| **支撑 25 个团队所需 H100 数** | `80 张 H100` ✕ | `40 张 H100` | `48 张 H100` | 🏆 **`24 张 H100`（砍掉 70% 冗余显卡规模）** |
| **月度平台总成本 (`TCO`)** | `$310,000 / 月` | `$178,000 / 月`（含故障损失） | `$164,000 / 月` | **`$82,000 / 月` 🏆（节省 73.5%，年省 $273 万）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（25 个内部研发与业务团队 / 85% 共享基础模型 + 15% 自研小模型与微调）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开多团队共享企业级模型服务中台（Internal MaaS Platform）定容计算器](../reports/enterprise-internal-maas-platform-multi-tenant-gpu-sharing-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-internal-maas-platform-multi-tenant-gpu-sharing-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection/run.py --check
  python -m unittest tests/test_enterprise_internal_maas_platform_multi_tenant_gpu_sharing_selection.py
  ```

### 5.2 核心量化对比图（全集群 GPU 利用率 % vs. 邻里干扰与 OOM 误伤率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Siloed Team GPU Clusters", "metric": "Cluster GPU Utilization (%)", "pct": 22.0 },
      { "tier": "Tier 1: Siloed Team GPU Clusters", "metric": "Noisy-Neighbor OOM Rate (%)", "pct": 0.0 },
      { "tier": "Tier 2: GPU Time-Slicing / MPS", "metric": "Cluster GPU Utilization (%)", "pct": 68.0 },
      { "tier": "Tier 2: GPU Time-Slicing / MPS", "metric": "Noisy-Neighbor OOM Rate (%)", "pct": 31.5 },
      { "tier": "Tier 3: Static Hardware MIG Only", "metric": "Cluster GPU Utilization (%)", "pct": 54.0 },
      { "tier": "Tier 3: Static Hardware MIG Only", "metric": "Noisy-Neighbor OOM Rate (%)", "pct": 0.0 },
      { "tier": "Tier 4: 3-Layer MaaS (API+MIG+Kueue)", "metric": "Cluster GPU Utilization (%)", "pct": 86.5 },
      { "tier": "Tier 4: 3-Layer MaaS (API+MIG+Kueue)", "metric": "Noisy-Neighbor OOM Rate (%)", "pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 achieves 86.5% GPU utilization with 0.0% noisy-neighbor OOM interference",
    "subtitle": "Comparison across 25 internal enterprise engineering and business teams",
    "encodings": {
      "x": "tier",
      "y": "pct",
      "group": "metric",
      "color": "metric"
    }
  }
}
```

---

## ⑥ 🛠️ 架构评审与生产落地核验清单（Production Readiness Checklist）

- [ ] **停止向各业务部门直接发放裸 GPU 虚拟机去各自重复部署通用大模型（`70B / 32B / Embedding / Reranker`）**：由内部 MaaS 中台集中部署高吞吐共享推理池，通过自助门户向业务团队发放带配额与成本中心标签的 `Virtual API Key`。
- [ ] **在共享大模型推理池强制开启 `Chunked Prefill`（分块预填充）与租户级 `GCRA` 令牌桶限流**：防止单一团队的超长上下文请求阻塞其他团队的首字响应（TTFT）。
- [ ] **在生产环境中禁用软件级 `GPU Time-Slicing` 多团队混跑，改用硬件级 `NVIDIA MIG`（`1g.10gb / 2g.20gb / 3g.40gb`）承载团队自研小模型**：实现显存控制器与计算单元的物理级故障隔离。
- [ ] **部署 `Kubernetes Kueue` 实现跨事业部联盟配额借用（Cohort Fair-Sharing）与优雅抢占**：消灭“A 部门显卡闲置空转、B 部门作业排队一天”的资源错配。
- [ ] **集成 `Node Problem Detector (NPD)` 与 DCGM 硬件健康巡检（自动识别 `Xid 79 / 双比特 ECC 错误`）**：发现故障显卡在 2 秒内自动 Cordon 摘除并迁移工作负载。
