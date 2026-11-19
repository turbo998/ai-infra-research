# 企业级大模型推理弹性伸缩（Autoscaling）、GPU 冷启动加速与云端突发溢出（Burst Spillover）架构选型指南：静态峰值常驻（Static 24x7 GPU）vs. 传统 K8s `GPU_UTIL HPA` vs. 单纯 `KEDA` 队列扩容 vs. `KEDA` 显存/队列双指标 + `KAITO/NVMe` 36 秒极速加载 + AI 网关云端 `PAYGO` 无缝溢出架构

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-llm-autoscaling-keda-kaito-serverless-gpu-cold-start-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、Kubernetes / AKS 平台工程负责人与 FinOps 成本治理团队的大模型推理弹性扩缩容与冷启动优化实战指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection/results.json`](../../experiments/enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection/results.json)。

![企业级大模型推理弹性伸缩、GPU 冷启动加速与云端突发溢出选型总结图](../../assets/svg/enterprise-llm-autoscaling-keda-kaito-serverless-gpu-cold-start-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在企业内部办公助手（Copilot）、客服中心与研发代码补全等核心场景中，流量呈现出极强的**“潮汐效应（Diurnal Peak-to-Trough Ratio 高达 `8.5 : 1`）”**——工作日上午 09:00–11:00 需要 **32 张 H100 GPU** 才能扛住并发，而到了晚上 20:00 以后和周末，90% 的显卡都在空转。架构师在设计 Kubernetes（`AKS / EKS / GKE`）推理弹性伸缩时，常踩三大深坑：
1. **静态按峰值 24×7 常驻（Tier 1，月均 GPU 利用率仅 `26.5%`）**：为了保证早高峰不卡顿，常年死扛 32 张 H100，每月白白烧掉 `$198,000`，其中 73.5% 的算力在夜间和周末空转浪费；
2. **直接照搬传统微服务 K8s HPA 监控 `CPU` 或 `GPU 计算利用率 (DCGM_FI_DEV_GPU_UTIL)`（Tier 2，早高峰超时率高达 `34.2%`，P99 TTFT 飙到 `18.40s`）**：大模型解码（Decode）是典型的**显存容量与显存带宽受限型**负载——当并发长文本涌入时，`vLLM` 的 **KV Cache 显存占用率已经飙到 `99%`**、排队队列塞满 80 个请求，但 GPU 的算术计算单元利用率（`GPU_UTIL`）却只有 **`52%`**！结果 K8s HPA 以为“负载还不高”拒不扩容，线上大面积超时；
3. **只配了 `KEDA` 却没解决 70B 模型“11 分钟冷启动”与“扩容空窗期兜底”（Tier 3，早高峰丢包率 `11.8%`）**：每次扩容一个 70B FP8（`72 GB` 权重）Pod，从拉取 15 GB 容器镜像到下载 72 GB 模型文件动辄耗时 `270s~680s`（4.5 到 11 分钟）。等新 Pod 终于启动好，早高峰的第一波洪峰早就把线上冲垮了！

在 **早高峰峰值 32 张 H100、70B FP8 模型（`72 GB` 权重）、昼夜潮汐比 `8.5:1`、目标 P99 首字延迟 `TTFT <= 1.50s`** 的企业基准下，四档弹性架构的量化对比如下：

| 方案档位 | 核心架构组合 | 新 70B Pod 扩容就绪耗时 | 5 倍突发洪峰超时/丢包率 | 24×7 集群平均 GPU 利用率 | 早高峰 P99 `TTFT` | 月度推理算力总成本 (`TCO`) | 架构师核心选型结论 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：静态峰值 24×7 常驻 (`Static 32x H100`)** | 32 张 H100 全天候不缩容，无自动弹性 | `0.0s` *(常驻不启停)* | `0.5%` | `26.5%` ✕ *(73%夜间闲置)* | `1.15s` | `$198,000 / 月` ✕ | ⛔ 财务与 FinOps 绝对无法接受的吞金兽 |
| **🟨 Tier 2：传统 K8s HPA (`CPU / GPU_UTIL` 指标)** | 监控 GPU 计算利用率 > 80% 扩容 + 普通 HTTP 拉镜像与权重 | `680.0s (11.3 min)` ✕ | `34.2%` ✕ | `48.0%` | `18.40s` ✕ *(KV满但HPA不动作)* | `$134,000 / 月` ✕ | ⛔ 大模型扩容头号反模式，必出 P0 故障 |
| **🟦 Tier 3：单纯 `KEDA` 队列/KV 指标 (无加速/无云端溢出)** | 监控 `vllm:gpu_cache_usage_perc` 扩容，但仍需 4.5 分钟加载权重 | `270.0s (4.5 min)` ✕ | `11.8%` ✕ *(倒在4.5分钟空窗期)* | `64.5%` | `6.80s` ✕ | `$102,000 / 月` | ⚠️ 选对了指标，但败给了模型加载速度 |
| **🟩 Tier 4：`KEDA` 双先行指标 + `KAITO/NVMe` 36s 极速加载 + 网关云端 `PAYGO` 溢出** | **`KEDA (KV Cache + 等待队列)` + OCI 镜像流式挂载 / 本地 NVMe 权重直通显存 (`36s`) + AI 网关秒级溢出至云端 Serverless API** | **`36.0s` 🏆** *(快 94.7%)* | **`0.0%` 🏆** *(云端兜底36秒空窗)* | **`82.4%` 🏆** | **`0.92s` 🏆** | **`$78,000 / 月` 🏆** *(-60.6%)* | 🌟 **企业级自建大模型弹性集群唯一标准答案** |

> 💡 **架构师一句话选型铁律**：**“给自托管大模型集群做弹性伸缩，必须完成‘三件套闭环’：① 彻底废除 `GPU_UTIL` 指标，改用 `KEDA` 监控 `vllm:gpu_cache_usage_perc > 75%` 与 `num_requests_waiting > 5` 先行指标；② 用容器镜像按需流式挂载（Artifact Streaming）+ 本地 NVMe 预热缓存并发加载（`KAITO / Run:ai Model Streamer`）把 70B 冷启动从 11 分钟压到 `36 秒`；③ 在新 Pod 启动的这 36 秒空窗期内，由 AI 网关将超额排队请求自动溢出（Spillover）到 Azure OpenAI / Bedrock Serverless PAYGO 按量接口！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大扩容失灵、冷启动慢与缩容断流痛点）

### Q1：为什么监控 `DCGM_FI_DEV_GPU_UTIL`（GPU 算力利用率）在 Web/训练时代很好用，到了大模型推理时代却会彻底失灵？应该监控什么指标？
- **大白话与第一性原理**：
  1. **Memory-Bound（访存与显存容量受限）的本质**：在大模型自回归生成（Decode）阶段，每生成一个 Token 都要把几十 GB 的模型权重和 KV Cache 从 HBM 显存搬运到计算单元，算术计算单元大部分时间都在“等显存喂数据”。因此即便单机并发已经把 **KV Cache 显存塞满到 `99%`**、新进来的请求全部被挂起排队（Waiting Queue 激增），`GPU_UTIL` 往往也只有 `45%~55%`，根本触发不了 `> 80%` 的传统 HPA 阈值！
  2. **Tier 4 的 `KEDA` 复合先行指标（Leading Indicators）**：
     - **指标一（显存水位）**：`vllm:gpu_cache_usage_perc > 0.75`（当 KV Cache 占用超过 75% 时提前触发扩容）；
     - **指标二（排队深度）**：`vllm:num_requests_waiting > 5`（一旦出现排队等待 Prefill 的请求立即拉起新副本）；
     - **指标三（定时潮汐预热）**：结合 `KEDA Cron Trigger`，在每个工作日上午 `08:45` 提前将基线副本数从夜间的 4 个预热拉升至 16 个！

### Q2：70B 大模型加上推理引擎镜像动辄近百 GB，怎么把新 Pod 从拉起到可服务（Ready）的时间从 11 分钟（`680s`）压缩到 `36 秒`？
- **消除网络重复拉取的“三段式极速冷启动流水线”**：
  1. **第一段（容器镜像 `4 秒` 就绪）**：开启 AKS Artifact Streaming / AWS SOCI（Seekable OCI），无需下载完整的 15 GB PyTorch+CUDA+vLLM 镜像即可秒级启动容器；
  2. **第二段（72 GB 模型权重 `24 秒` 直通显存）**：采用 **`KAITO` 预烘焙权重盘** 或在 GPU 节点本地 RAID-0 NVMe SSD / DaemonSet HostPath 上预热 Safetensors 权重，配合 **`Run:ai Model Streamer` / `Tensorizer`** 以多线程 Direct-I/O 绕过 CPU 内存拷贝、按 `12~16 GB/s` 带宽直接流式注入 H100 HBM3 显存；
  3. **第三段（CUDA Graph 预编译缓存 `8 秒`）**：持久化挂载 vLLM / TorchInductor 编译缓存目录，避免每次重启重新花费数分钟编译内核，总就绪时间稳定在 **`36.0 秒`**！

### Q3：就算冷启动压到了 36 秒，如果早高峰 09:00 瞬间涌入 5 倍流量，这前 36 秒里的请求岂不是照样要排队超时？晚间缩容时又怎么防止把正在流式输出的 60 秒长回答直接掐断？
- **“云端 Serverless 36 秒削峰溢出” + “零断流优雅缩容（Graceful Drain）”**：
  1. **AI 网关秒级突发溢出（Cloud Burst Spillover）**：当 AI 网关（如 Envoy AI Gateway / LiteLLM / APIM）检测到自建 vLLM 池的等待队列超过阈值而新 Pod 仍在 36 秒启动途中时，自动将溢出的 15%~20% 洪峰请求瞬时路由到云端 **Azure OpenAI / AWS Bedrock Serverless PAYGO** 接口。等 36 秒后自建 H100 新副本 Ready，流量自动切回自建集群——既实现了 **`0.0%` 洪峰丢包与 `0.92s` P99 TTFT**，又比全年常驻 32 张卡省下 **`60.6%`（每月立省 `$120,000`）**！
  2. **零断流优雅缩容**：为 vLLM Pod 配置 `terminationGracePeriodSeconds: 180` 与 `PreStop` 钩子——收到缩容信号后先从网关摘除流量，并轮询等待 `vllm:num_requests_running == 0` 后再销毁容器，确保长思考/Agent 流式输出 **`0.0%` 中途断流**。

---

## ③ 🧭 直观可视化图解（决策树 + KEDA/KAITO/云溢出参考架构图 + 36 秒极速扩容时序图）

### 3.1 企业级大模型推理弹性伸缩与冷启动架构选型决策树

```mermaid
flowchart TD
    A["企业级自建大模型推理集群弹性伸缩选型<br/>(峰值 32 张 H100 + 70B FP8 模型 + 昼夜流量比 8.5:1)"] --> B{"夜间与周末低谷期是否释放闲置 GPU?"}
    B -- "否 (24x7 常驻 32 张 H100)" --> C["⛔ 73.5% 算力空转浪费 (Tier 1)<br/>月度算力账单高达 $198,000/月"]
    B -- "是 (按需动态扩缩容)" --> D{"采用什么扩容指标与模型权重加载方案?"}
    D -- "监控 CPU 或 GPU_UTIL + 常规拉取镜像/权重" --> E["⛔ 扩容失灵与 11 分钟冷启动灾难 (Tier 2)<br/>KV 显存已满但 GPU_UTIL 仅 52%，早高峰超时率 34.2%"]
    D -- "监控 KEDA KV-Cache/队列 + 36s 极速加载 + 云端溢出" --> F["🟩 Tier 4: KEDA KV 指标 + KAITO 36s 加载 + 网关云端溢出 🏆<br/>(36s 极速就绪 | 0.0% 洪峰超时 | 82.4% 利用率 | 省 60.6% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `KEDA` 显存/队列双指标 + `KAITO/NVMe` 36 秒加载 + 网关云端溢出架构图

```mermaid
flowchart LR
    subgraph Gateway["1. AI 网关秒级削峰溢出层"]
        Req["早高峰 5x 突发请求流"] --> AIGW["🌐 AI Gateway (Envoy / APIM)<br/>优先路由自建 H100 池；当排队 > 5 时<br/>瞬时溢出至云端 Serverless PAYGO"]
    end

    subgraph Cluster["2. KEDA + KAITO 36s 极速弹性自建集群"]
        AIGW -->|主力流量| VLLM["🚀 自建 vLLM / SGLang 推理池<br/>暴露 gpu_cache_usage_perc 与队列指标"]
        VLLM -->|指标超阈值 (KV > 75%)| KEDA["📈 KEDA Autoscaler + Cron 预热<br/>立即触发副本扩容与优雅缩容"]
        KEDA --> FastBoot["⚡ 36s 极速冷启动流水线<br/>OCI 镜像流 (4s) + 本地 NVMe 权重直通显存 (24s)"]
        AIGW -.->|仅在 0s~36s 扩容空窗期溢出| Cloud["☁️ Azure OpenAI / Bedrock PAYGO<br/>0.0% 洪峰超时兜底"]
    end
```

### 3.3 早高峰 09:00 突发 5 倍洪峰：从“云端溢出兜底”到“36 秒新副本接管”全链路时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 企业员工早高峰请求 (5x Surge)
    participant GW as AI Gateway (含溢出路由)
    participant VLLM as 现有 vLLM 副本池
    participant KEDA as KEDA + KAITO 极速扩容控制器
    participant Cloud as 云端 Serverless PAYGO API

    User->>GW: t=0s：周一 09:00 流量瞬间暴涨 5 倍
    GW->>VLLM: 路由至现有 H100 副本，`gpu_cache_usage_perc` 突破 75%，等待队列 > 5
    VLLM->>KEDA: t=2s：Prometheus 暴露 KV 显存与等待队列先行指标，KEDA 立即触发扩容 8 个新副本
    GW->>Cloud: t=2s~38s（36 秒扩容空窗期）：网关自动将超额排队请求无缝溢出至云端 PAYGO（0.0% 丢包，P99 TTFT 0.92s！）
    KEDA->>VLLM: t=38s：新副本通过 OCI 流式挂载 + 本地 NVMe 并发直通显存完成 72GB 权重加载，进入 Ready 状态！
    GW->>VLLM: t=38s 之后：100% 流量切回自建低成本 H100 集群，全天平均 GPU 利用率达 82.4%！
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：静态峰值常驻 (`32x H100`) | 🟨 Tier 2：传统 K8s `GPU_UTIL HPA` | 🟦 Tier 3：单纯 `KEDA` (常规慢加载) | 🟩 Tier 4：`KEDA` + `36s 极速加载` + `云端溢出` |
|---|---|---|---|---|
| **扩容触发指标精准度** | 无弹性（全天固定 32 卡） | ✕ 极差（KV 满时 `GPU_UTIL` 仅 52%，不扩容） | 优（监控 KV Cache 与等待队列） | 🏆 **极优（KV Cache + 队列深度 + Cron 定时预热）** |
| **70B 模型新副本就绪耗时** | `0.0s`（常驻耗电） | `680.0s`（11.3 分钟）✕ | `270.0s`（4.5 分钟）✕ | 🏆 **`36.0s`（OCI 流式镜像 + NVMe 并发直通显存）** |
| **5 倍早高峰突发超时/丢包率** | `0.5%` | `34.2%` ✕ | `11.8%` ✕（死在 270 秒启动空窗期） | 🏆 **`0.0%`（AI 网关用云端 PAYGO 完美填平 36 秒空窗）** |
| **24×7 集群平均 GPU 利用率** | `26.5%` ✕（夜间/周末大量闲置） | `48.0%` | `64.5%` | 🏆 **`82.4%`（夜间优雅缩容至最小保底池）** |
| **月度推理算力总成本 (`TCO`)** | `$198,000 / 月` | `$134,000 / 月` | `$102,000 / 月` | **`$78,000 / 月` 🏆（节省 60.6%，立省 $120K/月）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（峰值 32 张 H100 / 70B FP8 `72 GB` 权重 / 昼夜潮汐比 `8.5:1`）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开企业级大模型推理弹性伸缩、冷启动加速与云端溢出定容计算器](../reports/enterprise-llm-autoscaling-keda-kaito-serverless-gpu-cold-start-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-llm-autoscaling-keda-kaito-serverless-gpu-cold-start-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection/run.py --check
  python -m unittest tests/test_enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection.py
  ```

### 5.2 核心量化对比图（24×7 平均 GPU 利用率 % vs. 5 倍洪峰超时丢包率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Static 24x7 32x H100", "metric": "24x7 GPU Utilization (%)", "pct": 26.5 },
      { "tier": "Tier 1: Static 24x7 32x H100", "metric": "5x Surge Timeout Rate (%)", "pct": 0.5 },
      { "tier": "Tier 2: Standard GPU_UTIL HPA", "metric": "24x7 GPU Utilization (%)", "pct": 48.0 },
      { "tier": "Tier 2: Standard GPU_UTIL HPA", "metric": "5x Surge Timeout Rate (%)", "pct": 34.2 },
      { "tier": "Tier 3: KEDA Slow Weight Pull", "metric": "24x7 GPU Utilization (%)", "pct": 64.5 },
      { "tier": "Tier 3: KEDA Slow Weight Pull", "metric": "5x Surge Timeout Rate (%)", "pct": 11.8 },
      { "tier": "Tier 4: KEDA + 36s Load + Spillover", "metric": "24x7 GPU Utilization (%)", "pct": 82.4 },
      { "tier": "Tier 4: KEDA + 36s Load + Spillover", "metric": "5x Surge Timeout Rate (%)", "pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 achieves 82.4% GPU utilization with 0.0% surge timeouts",
    "subtitle": "Comparison across 32x H100 peak diurnal workload (8.5:1 peak-to-trough ratio)",
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

- [ ] **严禁在大模型推理服务上使用 `CPU` 或 `DCGM_FI_DEV_GPU_UTIL` 作为主扩容指标**：统一通过 `KEDA` 采集推理引擎原生暴露的 `vllm:gpu_cache_usage_perc`（阈值 `70%~75%`）与 `vllm:num_requests_waiting`（阈值 `> 5`）。
- [ ] **启用容器镜像按需流式加载（AKS Artifact Streaming / AWS SOCI）与本地 NVMe 权重预热缓存**：使用 `KAITO` 或 `Run:ai Model Streamer` 多线程并发读取 Safetensors 权重直通 GPU 显存，并持久化挂载 CUDA Graph 编译缓存，确保 70B 新副本在 `<= 40 秒` 内 Ready。
- [ ] **在 AI 网关层配置“扩容空窗期云端 Serverless API 自动溢出（Spillover）”**：当自建池排队深度超标而新副本尚未 Ready 时，自动将溢出请求转发至 Azure OpenAI / Bedrock PAYGO，消除扩容等待期的 503/超时报错。
- [ ] **配置工作日早高峰 `KEDA Cron` 提前 15 分钟预热基线副本**：避免周一早上 09:00 从最低副本数被动冷启动。
- [ ] **配置 `terminationGracePeriodSeconds >= 180` 与基于 `num_requests_running == 0` 的 `PreStop` 优雅排空钩子**：确保晚间缩容时绝不中断正在进行的长文本流式生成或 Agent 工具调用。
