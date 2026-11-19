# 企业级多云竞价实例（Spot / Preemptible GPU）、原子成组供给（`Kueue ProvisioningRequest / GCP DWS / AWS Capacity Blocks`）、`SkyPilot` 跨区寻优与 `30s` 抢占优雅排空架构选型指南：单可用区按需实例（On-Demand）裸扩 vs. 裸跑单区 Spot 实例 vs. 单云多可用区 Karpenter Spot vs. `Kueue` 全有或全无原子供给 + `30% 基荷 / 70% 多池 Spot` + `IMDS 30s` 零丢包优雅排空架构

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-multi-cloud-spot-gpu-sky-dws-kueue-fallback-serving-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、多云基础设施与 FinOps 负责人、Kubernetes（`EKS / GKE / AKS`）智算平台架构师的云端稀缺 GPU 容量获取、Spot 抢占高可用与成本优化指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection/results.json`](../../experiments/enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection/results.json)。

![企业级多云 Spot GPU、Kueue DWS 原子供给与 SkyPilot 优雅排空架构总结图](../../assets/svg/enterprise-multi-cloud-spot-gpu-sky-dws-kueue-fallback-serving-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在企业规划 **96 张 H100 级别云端 GPU 算力池（承载在线推理弹性扩容与分布式微调任务）** 时，云端 GPU 既昂贵又经常面临单一可用区库存告罄（`InsufficientInstanceCapacity`）。若架构设计不当，企业会掉入三大云算力陷阱：
1. **传统 K8s Cluster Autoscaler 逐台拉起节点（Tier 1），在多机任务遇到库存不足时造成高达 `22.0%` 的“半吊子成组空转浪费（Partial-Gang Allocation Deadlock）”**：一个需要 `2 台 8×H100（共 16 卡）` 协同启动的训练/大模型推理任务，云厂商当时只剩 1 台现货。传统自动扩缩容先把第 1 台拉起来（每小时烧掉 `$32`），然后傻等第 2 台长达 4 小时——期间任务根本无法启动，已拉起的 8 张 H100 100% 空转烧钱！
2. **裸跑单可用区 Spot 竞价实例（Tier 2）未对接云厂商 `IMDS 30s–120s` 抢占预警钩子，回收瞬间 `100.0%` 正在输出的 SSE 推理流直接报错 `HTTP 502` 断开，且遭遇同区批量回收风暴时缺货率高达 `34.0%`**！
3. **跨云/跨区突发扩容时每次从主区公网拉取 140GB 模型权重，产生高昂的跨云公网流量费（Egress Fee）与长达 7 分钟的冷启动**！

在 **96 卡 H100 等效规模、横跨 4 个云区域/机型池** 的企业基准负载下，四档多云 GPU 容量供给与 Spot 高可用架构的量化对决如下：

| 方案档位 | 核心架构组合 | 云端 GPU 缺货分配失败率 (`Stockout`) | Spot 抢占在途请求断流率 | 多机成组半拉起空转浪费率 | 抢占切换与补齐恢复耗时 | 月度云端 GPU 算力总成本 (`TCO`) | 生产推荐度 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：单可用区 100% 按需实例 (`On-Demand`) + 默认 Autoscaler** | 锁死单 AZ 按需实例 + 逐节点扩容（无原子成组预约） | `28.4%` ✕ *(单区H100常缺货)* | `18.5%` ✕ *(缩容粗暴杀Pod)* | `22.0%` ✕ *(开出1台等第2台)* | `420.0 秒` ✕ | `$215,000 / 月` ✕ | ⛔ 既贵又常买不到卡，半拉起白烧钱 |
| **🟨 Tier 2：单区裸跑 Spot 竞价实例 (无 `IMDS` 排空与跨池兜底)** | 单 AZ Spot 直接跑推理/训练，等 `SIGKILL` 硬杀 | `34.0%` ✕ *(同区连片被收回)* | `100.0%` ✕ *(在途流全断)* | `19.5%` ✕ | `510.0 秒` ✕ | `$142,000 / 月` ✕ | ⛔ 虽省单价但线上频繁 502 断流 |
| **🟦 Tier 3：单云多可用区 Karpenter / NAP Spot 池** | 单云跨 AZ 混用 Spot，但缺 `Kueue` 原子成组供给与跨云/跨代兜底 | `9.2%` | `4.8%` ✕ | `8.4%` ✕ | `135.0 秒` | `$118,000 / 月` ✕ | ⚠️ 整个大区同款 GPU 售罄时仍会卡死 |
| **🟩 Tier 4：`Kueue ProvisioningRequest` (`DWS`) + `SkyPilot` 跨池 + `30s` 排空** | **全有或全无原子供给 + `30% 按需基荷 / 70% 四池异构 Spot` + `IMDS 30s` 网关摘流排空 + 零出网费 R2/P2P 镜像** | **`0.1%` 🏆** *(-28.3 pts)* | **`0.0%` 🏆** *(零在途断流)* | **`0.0%` 🏆** *(零半吊子空转)* | **`14.0 秒` 🏆** *(-96.7%)* | **`$79,000 / 月` 🏆** *(-63.3%)* | 🌟 **企业级高性价比弹性 GPU 云唯一标准** |

> 💡 **架构师一句话选型铁律**：**“企业上云用 GPU 决不能死磕单一可用区的按需实例，更不能把多机任务拆成单台 VM 盲目去碰运气——必须启用 Kubernetes `Kueue ProvisioningRequest` 对接云厂商原子容量接口（`GCP DWS / AWS Capacity Blocks`）实现‘全有或全无（All-or-Nothing）’成组开机；用 `30% 按需/预留保底基荷 + 70% 跨可用区/跨机型等价池（H100 ↔ A100/L40S）Spot 竞价实例’对冲缺货；并通过监听云主机 `IMDS 30s–120s` 抢占通知在 `t=0s` 从 Envoy 网关摘除节点、排空在途 SSE 流！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大云端 GPU 缺货、半吊子空转与 Spot 闪断死穴）

### Q1：为什么 16 卡/32 卡分布式任务用默认 K8s Cluster Autoscaler 扩容时，会白白浪费 `22.0%` 的云账单？`Kueue ProvisioningRequest`（`DWS`）怎么做到 `0.0%` 浪费？
- **大白话与第一性原理**：
  1. **云端节点级扩容与作业级成组调度（Gang Scheduling）的割裂**：当提交一个需要 `2 台 8×H100` 节点的 PyTorch FSDP 微调或 `DeepSeek 671B` 双机推理任务时，传统 Cluster Autoscaler 向云 API 请求 2 台机器。如果云机房当时只有 1 台空闲，云厂商会先交付并启动第 1 台（立刻开始按 `$32/小时` 计费），而第 2 台一直报 `InsufficientInstanceCapacity`。由于分布式任务缺 1 台就无法初始化 `NCCL` 通信环，第 1 台机器在集群里白白空转数小时！
  2. **Tier 4 的“全有或全无原子供给（Atomic All-or-Nothing `ProvisioningRequest`）”**：
     - 通过 Kubernetes `Kueue` 创建 `ProvisioningRequest` CRD，底层直连 **Google Cloud `DWS`（Dynamic Workload Scheduler）、AWS `EC2 Capacity Blocks / Instant Capacity`** 或 **Azure `AKS Node Autoprovisioning`** 的原子批量接口；
     - 云厂商在后台排队凑齐全部 16 张同拓扑 GPU 之前，**绝不提前开机计费任何一台半成品节点**，将成组半拉起空转浪费从 `22.0%` 彻底清零（**`0.0%`**）！

### Q2：Spot 竞价实例价格只有按需实例的 `30%~40%`，但云厂商随时只给 `30 秒（GCP/Azure）到 120 秒（AWS）` 通知就会强行断电，在线大模型推理怎么做到 `0.0%` 掉线？
- **`IMDS` 抢占元数据亚秒级拦截 + Envoy 网关瞬时摘流 + 在途流分块续写**：
  1. 每个 GPU 节点运行轻量级守护进程（如 `aws-node-termination-handler` 或自研 `IMDS Watcher`），每 `500ms` 轮询本机 `http://169.254.169.254` 抢占事件；
  2. **第 `0.1 秒`（瞬时摘流）**：一旦收到抢占预警，立即将 Pod 标记为 `Draining` 并通知上游 Envoy AI Gateway / `EPP` 停止向该节点派发任何新请求；
  3. **第 `0.5 ~ 14 秒`（在途流优雅排空与续写）**：对正在 Decode 的活跃流，剩余生成词元少于 300 个的在 `14 秒` 内自然输出完毕；对于超长思维链请求，网关记录已下发给客户端的 Token 序列并在温备节点上触发 Prefix 续写（Mid-Stream Resume），实现 **`0.0%` 用户侧报错断流**！

### Q3：如果云厂商某个可用区爆发“Spot 批量回收风暴”，同时收走 80% 的 H100 Spot 怎么办？跨云/跨区扩容如何免除天价流量费？
- **四池异构解耦（Multi-AZ × Multi-SKU Equivalence）+ 零出网费对象存储（`Cloudflare R2`）预热**：
  1. **不把鸡蛋放在同一个可用区和同一款型号里**：若单池缺货率为 $p = 34\%$，通过 `SkyPilot / Karpenter` 同时横跨 **4 个互不相关的故障域与等价机型池**（例如 Pool 1: `us-east-1a 8×H100`、Pool 2: `us-west-2b 8×H100`、Pool 3: `us-east-4c 8×A100-80GB`、Pool 4: `2×8 L40S FP8`），同时缺货的联合概率骤降至 $p^4 \approx 0.1\%$！
  2. **零跨云公网 Egress 架构**：绝不在扩容时跨公网拉取 140GB 模型权重——所有区域预先通过 **零出网流量费的 Cloudflare R2（或各云本地域内 S3/GCS 桶）+ VPC 内网 `Dragonfly P2P`** 缓存只读 Safetensors 权重，跨云仅传输 `4KB` 的 JSON 文本请求！

---

## ③ 🧭 直观可视化图解（决策树 + 原子供给/跨池调度架构图 + 30秒抢占零丢包时序图）

### 3.1 企业级多云 GPU 算力采买与 Spot 高可用架构选型决策树

```mermaid
flowchart TD
    A["企业级云端 GPU 算力采买与弹性推理/微调集群<br/>(96 卡 H100 等效规模 + 跨区域弹性伸缩)"] --> B{"是否锁死单可用区按需实例 (On-Demand)<br/>或直接裸跑无排空保护的单区 Spot 实例?"}
    B -- "是 (单区按需或裸跑 Spot)" --> C["⛔ 缺货卡死、半拉起烧钱或 100% 闪断 (Tier 1 / Tier 2)<br/>28.4%~34% 缺货失败，22% 半拉起空转浪费，或 Spot 回收时在途请求全断"]
    B -- "引入多可用区 Spot 池" --> D{"是否启用 Kueue ProvisioningRequest 原子成组供给、<br/>跨区/跨机型等价池与 IMDS 30s 网关优雅排空?"}
    D -- "否 (仅用单云普通多 AZ Spot)" --> E["⚠️ 同型号全区售罄与残余断流 (Tier 3)<br/>仍有 9.2% 缺货率、8.4% 成组等待浪费与 4.8% 抢占断流"]
    D -- "Kueue DWS 原子供给 + 四池异构 Spot + 30s 排空" --> F["🟩 Tier 4: Kueue DWS 原子供给 + SkyPilot 四池 Spot + 30s 排空 🏆<br/>(0.1% 缺货率 | 0.0% 抢占断流 | 0.0% 成组空转 | 省 63.3% GPU TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `Kueue ProvisioningRequest` (`DWS`) + `SkyPilot` 四池 Spot + `30s` 排空架构图

```mermaid
flowchart LR
    subgraph Control["1. 原子成组与跨池寻优控制面"]
        Job["分布式训练 / 弹性推理池"] --> Kueue["🧩 Kueue ProvisioningRequest<br/>对接 GCP DWS / AWS Capacity Blocks<br/>全有或全无原子分配 (0.0% 半拉起浪费)"]
        Job --> Sky["🌐 SkyPilot / Karpenter 跨池 Broker<br/>30% 按需基荷 + 70% 四池异构 Spot<br/>(H100 ↔ A100-80G ↔ L40S 等价切换)"]
    end

    subgraph Drain["2. IMDS 30s–120s 抢占优雅排空闭环"]
        IMDS["⚡ 云主机 IMDS 30s 抢占通知"] --> Hook["🛡️ 节点终止守护进程 (t=0.1s)"]
        Hook -->|1. 瞬时摘除端点| Envoy["Envoy AI Gateway<br/>(新请求切往温备池 + 长流断点续写)"]
        Hook -->|2. 14s 排空在途流| Pod["vLLM Pod 优雅退出 (0.0% 报错)"]
    end

    subgraph Storage["3. 零公网出网费权重预热"]
        R2["📦 Cloudflare R2 (零 Egress) / 本地桶<br/>+ 区域内 Dragonfly P2P 极速加载"] --> Pod
    end
```

### 3.3 从“收到云厂商 30 秒 Spot 抢占预警”到“14 秒零丢包排空并跨池补齐”时序图

```mermaid
sequenceDiagram
    autonumber
    participant Cloud as 云厂商控制面 (IMDS / DWS)
    participant Node7 as Spot GPU 节点 #7 (IMDS Watcher)
    participant Envoy as Envoy AI Gateway (全局路由)
    participant PoolB as 备选等价池 (Region-B / A100-80G)

    Cloud->>Node7: t = 0.0s: 下发 Spot 实例 30 秒后回收预警 (HTTP 169.254.169.254/preempted)
    Node7->>Envoy: t = 0.1s: 立即标记 Pod 为 Draining，从 InferencePool 摘除 Node #7（新请求 100% 零进入）
    Envoy->>PoolB: t = 0.2s: 将增量流量平滑路由至 Region-B 活跃副本，并触发 SkyPilot 在备选池补齐 1 台 Spot 节点
    Node7->>Node7: t = 0.2s–13.8s: 继续算完本机剩余 18 个在途 SSE 流的尾词；对 1 个超长流触发 Envoy 前缀续写移交
    PoolB-->>Envoy: t = 14.0s: 备选池新 Spot 节点通过本地 Dragonfly P2P 加载完权重就绪上线！
    Cloud->>Node7: t = 30.0s: 云厂商回收物理虚拟机 -> 此时 Node #7 在途请求早已清零 (0.0% HTTP 502 断流!)
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：单区按需 + 默认 Autoscaler | 🟨 Tier 2：单区裸跑 Spot (无排空) | 🟦 Tier 3：单云多 AZ Spot 池 | 🟩 Tier 4：`Kueue DWS` + 四池 Spot + `30s` 排空 |
|---|---|---|---|---|
| **云端 GPU 缺货分配失败率 (`Stockout`)** | `28.4%` ✕（单区 H100 常售罄） | `34.0%` ✕（同区批量回收） | `9.2%` | 🏆 **`0.1%`（4 个非相关区域/等价机型池联合寻优）** |
| **Spot 抢占在途请求断流率** | `18.5%` ✕（缩容未等长流结束） | `100.0%` ✕（`SIGKILL` 硬断连） | `4.8%` ✕ | 🏆 **`0.0%`（`IMDS t=0s` 网关摘流 + 在途流续写排空）** |
| **多机成组半拉起空转浪费率** | `22.0%` ✕（开出 1 台傻等第 2 台） | `19.5%` ✕ | `8.4%` ✕ | 🏆 **`0.0%`（`Kueue ProvisioningRequest` 全有或全无原子供给）** |
| **抢占切换与容量补齐耗时** | `420.0 秒` ✕ | `510.0 秒` ✕ | `135.0 秒` | 🏆 **`14.0 秒`（温备切换 + 本地 R2/Dragonfly P2P 秒级拉起）** |
| **月度云端 GPU 算力总成本 (`TCO`)** | `$215,000 / 月` | `$142,000 / 月` | `$118,000 / 月` | **`$79,000 / 月` 🏆（节省 63.3% 云算力账单）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（96 卡 H100 等效集群 / 30% 按需基荷 + 70% Spot / 4 个区域与机型池）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级多云 Spot GPU、Kueue DWS 与 SkyPilot 容灾定容计算器](../reports/enterprise-multi-cloud-spot-gpu-sky-dws-kueue-fallback-serving-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-multi-cloud-spot-gpu-sky-dws-kueue-fallback-serving-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection/run.py --check
  python -m unittest tests/test_enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection.py
  ```

### 5.2 核心量化对比图（云端 GPU 缺货失败率 % vs. 多机成组半拉起空转浪费率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Single-AZ On-Demand", "metric": "GPU Stockout Failure (%)", "pct": 28.4 },
      { "tier": "Tier 1: Single-AZ On-Demand", "metric": "Partial-Gang Idle Waste (%)", "pct": 22.0 },
      { "tier": "Tier 2: Raw Single-Zone Spot", "metric": "GPU Stockout Failure (%)", "pct": 34.0 },
      { "tier": "Tier 2: Raw Single-Zone Spot", "metric": "Partial-Gang Idle Waste (%)", "pct": 19.5 },
      { "tier": "Tier 3: Multi-AZ Spot Only", "metric": "GPU Stockout Failure (%)", "pct": 9.2 },
      { "tier": "Tier 3: Multi-AZ Spot Only", "metric": "Partial-Gang Idle Waste (%)", "pct": 8.4 },
      { "tier": "Tier 4: Kueue DWS + 4-Pool Spot", "metric": "GPU Stockout Failure (%)", "pct": 0.1 },
      { "tier": "Tier 4: Kueue DWS + 4-Pool Spot", "metric": "Partial-Gang Idle Waste (%)", "pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 cuts GPU stockout failures to 0.1% and eliminates partial-gang idle waste (0.0%)",
    "subtitle": "Comparison across a 96-GPU H100-equivalent enterprise cloud fleet (lower is better)",
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

- [ ] **所有多节点分布式训练与多机推理任务必须通过 `Kueue ProvisioningRequest` 请求“全有或全无（All-or-Nothing）”原子容量**：在云厂商凑齐全部所需 GPU 节点前严禁提前启动并计费单台孤立节点。
- [ ] **在线推理集群实行“`30%` 按需/预留实例保底基荷 + `70%` 跨可用区/跨机型等价池 Spot 弹性实例”混部**：预定义异构等价映射（如 `8×H100` ↔ `8×A100-80GB` ↔ `2×8 L40S`），防范单一型号全区断货。
- [ ] **每个 Spot GPU 节点必须运行 `IMDS` 抢占预警守护进程并与 Envoy AI 网关直连**：在收到 `30s–120s` 抢占通知的第 `0.1 秒` 立即摘除流量，并开启在途流排空与长流断点续写。
- [ ] **分布式训练/微调任务在 Spot 池运行时，将 `30s` 抢占信号绑定至“Host Pinned DRAM 两阶段秒级异步 Checkpoint”**：确保在 `30 秒` 断电宽限期内 100% 完成步级状态落盘。
- [ ] **跨云/跨区域弹性节点严禁跨公网拉取百 GB 模型镜像与权重**：必须在各目标区域部署零出网费对象存储（如 Cloudflare R2 / 同区 S3）与 VPC 内网 `Dragonfly P2P` 预热缓存。
