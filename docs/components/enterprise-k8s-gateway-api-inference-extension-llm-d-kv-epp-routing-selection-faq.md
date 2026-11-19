# 企业级 Kubernetes Gateway API Inference Extension（`InferencePool` + `EPP`）、`llm-d` 全局 KV/LoRA 感知路由与优先级分级限流（Criticality Shedding）架构选型指南：传统 K8s `Service` 轮询 vs. HTTP 最小连接数（Least-Connections）vs. 客户端一致性哈希粘性会话 vs. Envoy `ext_proc` 端点选择器（`EPP`）+ `KV Block-Hash / LoRA` 复合打分 + 85% 水位三级优先级熔断架构

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-k8s-gateway-api-inference-extension-llm-d-kv-epp-routing-selection-faq.en.md)  
> **定位**：面向云与 AI 解决方案架构师（Solution Architect）、Kubernetes 平台工程负责人（`AKS / EKS / GKE / OpenShift`）、企业级大模型推理平台（`vLLM / SGLang / llm-d`）架构师的云原生流量治理与容量规划指南。  
> **数据口径（Provenance）**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`），全部定容模型封存于 [`experiments/enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection/results.json`](../../experiments/enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection/results.json)。

![企业级 Kubernetes Gateway API Inference Extension 与 llm-d KV/LoRA 感知路由架构总结图](../../assets/svg/enterprise-k8s-gateway-api-inference-extension-llm-d-kv-epp-routing-selection-faq-summary.svg)

---

## ① 📌 核心选型结论速览（Executive Summary — 3 分钟看懂怎么选）

在 **64 张 H100 (80GB) GPU、16 个 `vLLM / SGLang` 推理副本（Pods）、承载 `1,200 RPS` 混合长短上下文请求与 `48` 个部门级微调 LoRA 适配器** 的企业级 Kubernetes 生产集群中，沿用传统微服务网关（Nginx / 标准 Envoy / `kube-proxy`）会遭遇三大致命架构失配：
1. **HTTP 最小连接数（Least-Connections）完全看不见“请求 Token 长度 100 倍不对称”与“GPU HBM KV 显存饱和度”**：Pod A 上虽然只有 `2` 个连接，但都是 `64K` 超长 RAG 请求（HBM KV 显存已占满 `94%`）；Pod B 上有 `8` 个 `256 Token` 短对话（HBM KV 仅占 `12%`）。传统最小连接负载均衡（Tier 2）会把新来的 `32K` 大请求硬塞给连接数更少的 Pod A，当场打爆 HBM 触发 KV Cache Preemption（被迫踢出并重算正在生成的请求，导致 **`8.6%~14.8%` 的请求被抢占中断，P99 TTFT 暴涨至 `3,120–4,480 ms`**）！
2. **轮询（Round-Robin）彻底击穿跨轮次 Prefix Cache 与多 LoRA 驻留亲和性（命中率仅 `16.5%`），而死板的一致性哈希粘性路由（Tier 3）又会在大租户并发突发时造成单 Pod 排队雪崩（`P99 TTFT = 1,450 ms`）**！
3. **无优先级准入控制导致批量离线任务冲垮实时客服 SLA**：当后台文档摘要批处理任务涌入占满 `100%` KV 显存时，高优先级实时 Copilot 请求遭遇 **`22.4%` 的 SLO 违约**！

在 **64 卡 H100、16 个推理副本、1,200 RPS、48 个 LoRA 适配器** 的企业基准规模下，四档 Kubernetes 推理接入与路由架构的量化对决如下：

| 方案档位 | 核心架构组合 | P99 首字延迟 (`TTFT`) | Prefix Cache 与 LoRA 命中率 | HBM KV 抢占驱逐/OOM 率 | 核心业务 (`Critical`) SLO 违约率 | 达标月度 GPU 集群总成本 (`TCO`) | 生产推荐度 |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1：标准 K8s `Service` / L4-L7 轮询 (`Round-Robin`)** | 默认 `kube-proxy` 或无状态 Ingress 均匀轮询分发 | `4,480 ms` ✕ | `16.5%` ✕ *(8倍重复预填充)* | `14.8%` ✕ *(频繁KV重算)* | `22.4%` ✕ | `$148,000 / 月` ✕ | ⛔ 把有状态 KV 引擎当无状态 Web 切流 |
| **🟨 Tier 2：HTTP 最小连接数 (`Least-Conn`) + CPU HPA** | 按活跃 TCP 流数量路由 + 传统 CPU/内存指标扩缩容 | `3,120 ms` ✕ | `21.0%` ✕ | `8.6%` ✕ *(64K大请求误塞满卡)* | `15.2%` ✕ | `$126,000 / 月` ✕ | ⛔ TCP 连接数无法反映 Token 与显存负载 |
| **🟦 Tier 3：客户端一致性哈希粘性路由 (`Sticky Hash`)** | 按租户/会话 ID 静态哈希锁死固定 Pod | `1,450 ms` | `64.0%` | `5.4%` ✕ *(热点Pod打爆)* | `6.8%` ✕ | `$92,000 / 月` ✕ | ⚠️ 热点租户突发时单 Pod 严重排队 |
| **🟩 Tier 4：K8s `Gateway API Inference Extension` (`EPP`) + `llm-d` 复合打分** | **`InferencePool` + Envoy `ext_proc` 端点选择器 (`EPP`) 实时 KV/队列/前缀/LoRA 复合打分 + `InferenceModel` 85% 水位三级优先级熔断** | **`340 ms` 🏆** *(-92.4%)* | **`89.4%` 🏆** *(+72.9 pts)* | **`0.0%` 🏆** *(零抢占驱逐)* | **`0.1%` 🏆** *(核心零受损)* | **`$54,000 / 月` 🏆** *(-63.5%)* | 🌟 **云原生 K8s 大模型推理唯一生产标准** |

> 💡 **架构师一句话选型铁律**：**“在大模型推理集群中，1 个 64K 请求的显存与算力消耗等于 120 个短请求，且每个 Pod 的 HBM 里驻留着价值连城的 Prefix KV Cache 与 LoRA 权重——绝不能用无状态的 K8s `Service` 轮询或最小连接数切流；必须部署 Kubernetes 官方 `Gateway API Inference Extension`（`InferencePool` + `InferenceModel` + Envoy `ext_proc` 端点选择器 `EPP`），融合实时 `KV 显存占用率 + 等待队列深度 + 前缀块哈希/LoRA 驻留亲和度` 动态打分选点，并在集群 KV 水位达 `85%` 时自动拦截 `Sheddable` 批处理流量！”**

---

## ② 🎯 客户与架构师灵魂拷问 Q&A（直击 3 大 K8s 推理切流与扩缩容死穴）

### Q1：为什么微服务时代百试百灵的“HTTP 最小连接数（Least-Connections）+ CPU/内存 HPA”在大模型推理上会彻底失灵？
- **大白话与第一性原理**：
  1. **连接数 ≠ GPU 显存与算力负载**：在传统 Web 服务中，每个 HTTP 请求耗时相近；但在大模型服务中，Pod A 处理 2 个 `64K Token` 的财报分析请求时，GPU HBM 的 PagedAttention KV 块已经被占满 **`94%`**；而 Pod B 处理 8 个 `256 Token` 的简短闲聊，KV 显存仅占 **`12%`**。HTTP 最小连接数负载均衡看到 `2 < 8`，把下一个 `32K` 请求继续发往 Pod A——瞬间触发 `vLLM` 的 **`SequenceGroup Preemption`（被迫把正在 Decode 的长请求踢出显存并在稍后全量重算 Prefill）**，导致 P99 首字延迟飙升至 `3,120 ms`！
  2. **GPU 满载时 Pod CPU 利用率往往只有 `15%`**：大模型推理由 GPU HBM 带宽和 Tensor Core 瓶颈主导，K8s 原生 CPU/内存 HPA 直到请求全部超时都根本不会触发扩容。Tier 4 改为直接采集 `vLLM / SGLang` 暴露的 **`kv_cache_usage_perc` 与 `num_requests_waiting`** 驱动 `EPP` 路由与 KEDA 弹性扩缩容。

### Q2：如何既拿到“跨轮次 Prefix Cache 与多 LoRA 89.4% 的极高命中率”，又不会像静态粘性哈希（Tier 3）那样把单个热门 Pod 排爆？
- **`EPP`（Endpoint Picker）多维加权复合打分公式（`llm-d` / Gateway API Inference Extension）**：
  - Tier 4 通过 Envoy 的 `ext_proc`（External Processing）gRPC 钩子，在转发每个请求前由 **`EPP`（Endpoint Picker）** 对 `InferencePool` 内的候选 Pod 进行亚毫秒级打分：
    $$\text{Score}(p, r) = w_{\text{prefix}} \cdot \text{PrefixBlockMatch}(p, r) + w_{\text{lora}} \cdot \mathbb{I}(\text{LoRA}_r \in p) - w_{\text{kv}} \cdot \left(U_{\text{KV}}(p) + \frac{\text{InFlightTokens}(p)}{C_{\text{KV}}(p)}\right) - w_{\text{q}} \cdot Q_{\text{wait}}(p)$$
  - 当某个热门前缀或 LoRA 所在的 Primary Pod 等待队列 $Q_{\text{wait}}(p) > 5$ 或实时 KV 占用（含 `EPP` 本地记账的在途未落盘 Token）超过 `80%` 时，负向惩罚项自动压过前缀加分，平滑溢出至次优温备 Pod（将缓存命中率推至 **`89.4%`** 的同时实现 **`0.0%` KV 抢占丢包**）！

### Q3：当多个业务部门共享同一个 `InferencePool` 底座时，如何防止后台跑批任务把前台实时客服 Copilot 拖死？
- **`InferenceModel` CRD 声明式三级优先级（`Critical` / `Standard` / `Sheddable`）水位线熔断**：
  - 在 Kubernetes `InferenceModel` 资源对象中，将同一底座模型映射为不同业务入口并绑定优先级（例如 `copilot-live -> Critical`，`internal-kb -> Standard`，`nightly-doc-summary -> Sheddable`）；
  - 当整个 `InferencePool` 的聚合 KV 显存水位突破 **`85%` 警戒线**时，`EPP` 立即对 `Sheddable` 级别请求返回 `HTTP 429 (Shed)` 引导其进入异步队列退避，将最后 `15%` 的宝贵 HBM KV 缓冲100% 留给 `Critical` 实时流量（使核心业务 SLO 违约率从 `22.4%` 降至 **`0.1%`**）！

---

## ③ 🧭 直观可视化图解（决策树 + `InferencePool / EPP` 拓扑图 + 请求路由时序图）

### 3.1 企业级 Kubernetes 大模型推理网关与流量路由选型决策树

```mermaid
flowchart TD
    A["企业级 Kubernetes 大模型推理集群接入层<br/>(64xH100 + 16 个 vLLM/SGLang 副本 + 1,200 RPS + 48 个 LoRA)"] --> B{"是否仍使用默认 K8s Service 轮询<br/>或传统 HTTP 最小连接数 (Least-Connections)?"}
    B -- "是 (无状态 L4/L7 切流)" --> C["⛔ KV 抢占雪崩与缓存击穿 (Tier 1 / Tier 2)<br/>看不见 Token 长度与 HBM KV 饱和，16.5% 缓存命中，P99 TTFT 高达 3.1s~4.48s"]
    B -- "引入会话/租户路由" --> D{"是否采用 K8s Gateway API Inference Extension (EPP)<br/>综合评估 KV 占用、等待队列、前缀/LoRA 亲和与三级优先级?"}
    D -- "否 (仅用客户端静态一致性哈希)" --> E["⚠️ 热点 Pod 排队拥塞 (Tier 3)<br/>大租户突发时单 Pod 队列堆积，P99 TTFT 仍达 1,450 ms 且无批处理水位隔离"]
    D -- "EPP 复合打分 + 85% 水位分级限流" --> F["🟩 Tier 4: K8s Gateway API Inference Extension + llm-d EPP 🏆<br/>(340 ms P99 TTFT | 89.4% 前缀/LoRA 命中 | 0.0% KV 抢占 | 省 63.5% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `Gateway API Inference Extension` (`InferencePool` + `InferenceModel` + `EPP`) 架构图

```mermaid
flowchart LR
    subgraph CRDs["1. 声明式模型与流量分级 (InferenceModel)"]
        M1["🟢 Critical: 实时客服 Copilot"]
        M2["🔵 Standard: 内部知识助手"]
        M3["🟡 Sheddable: 批量文档摘要"]
    end

    subgraph Gateway["2. Envoy Gateway + ext_proc EPP 调度器"]
        M1 & M2 & M3 --> Envoy["🌐 Envoy AI Gateway"]
        Envoy <-->|gRPC ext_proc| EPP["🧠 Endpoint Picker (EPP)<br/>1. 85% KV 水位拦截 Sheddable<br/>2. 复合打分: PrefixHash + LoRA - KV% - Queue<br/>3. 本地 In-Flight Token 实时预扣"]
    end

    subgraph Pool["3. InferencePool (16x vLLM / SGLang 副本)"]
        EPP -->|精准直达最优 Pod IP| PodA["Pod A (KV: 94% -> 避开)"]
        EPP -->|命中 24K 前缀 + LoRA| PodB["🏆 Pod B (KV: 28%, LoRA 热驻留)"]
        PodA & PodB -.->|Prometheus / gRPC 指标流| EPP
    end
```

### 3.3 从“识别 64K 满载陷阱”到“精准命中 24K 共享前缀与热驻留 LoRA”时序图

```mermaid
sequenceDiagram
    autonumber
    participant App as 企业法务 Copilot 客户端
    participant Envoy as Envoy Gateway (Gateway API)
    participant EPP as Endpoint Picker (EPP 扩展)
    participant PodA as vLLM Pod A (2个64K长请求, KV=94%)
    participant PodB as vLLM Pod B (8个短请求, KV=12%, 含法务LoRA)

    App->>Envoy: POST /v1/chat/completions (model="lora-legal-v3", 32K Prompt 含 24K 共享合同模板)
    Envoy->>EPP: ext_proc gRPC 请求头与 Body 解析 (criticality="Critical", prefix_hash, target_lora)
    EPP->>EPP: 1. 过滤掉 Pod A (KV=94% > 85% 安全阈值，防止触发 HBM 抢占重算!)
    EPP->>EPP: 2. 计算 Pod B 复合分: 命中 24K 前缀块 (+0.45) + lora-legal-v3 已驻留 HBM (+0.30) - KV 12% (-0.06) = 最高分!
    EPP->>EPP: 3. 本地预扣 Pod B 在途 Token 账本 (+8K 增量 tokens)，防止 250ms 采集间隔内并发踩踏
    EPP-->>Envoy: 返回目标端点 x-gateway-destination-endpoint: Pod B (10.244.3.18:8000)
    Envoy->>PodB: 转发请求至 Pod B -> 直接复用 24K Prefix KV Cache 与热驻留 LoRA (TTFT = 340 ms!)
```

---

## ④ 📊 架构师四维选型决策矩阵（Architecture Decision Matrix）

| 评估维度 | 🟥 Tier 1：标准 K8s `Service` 轮询 | 🟨 Tier 2：HTTP 最小连接数 + CPU HPA | 🟦 Tier 3：客户端一致性哈希粘性路由 | 🟩 Tier 4：`Gateway API Inference Extension` (`EPP`) |
|---|---|---|---|---|
| **P99 首字响应延迟 (`TTFT`)** | `4,480 ms` ✕ | `3,120 ms` ✕ | `1,450 ms` | 🏆 **`340 ms`（降幅 `92.4%`）** |
| **Prefix Cache 与 LoRA 命中率** | `16.5%` ✕（跨 Pod 随机打散） | `21.0%` ✕ | `64.0%`（但无法动态溢出） | 🏆 **`89.4%`（KV 前缀块哈希 + LoRA 位图亲和 + 队列溢出）** |
| **HBM KV 抢占驱逐/OOM 率** | `14.8%` ✕（频繁踢出重算） | `8.6%` ✕（大 Prompt 误塞满卡） | `5.4%` ✕（热点 Pod 爆仓） | 🏆 **`0.0%`（实时 KV 水位感知 + 本地以 Token 计费预扣）** |
| **核心业务 (`Critical`) SLO 违约率** | `22.4%` ✕（被批处理挤占） | `15.2%` ✕ | `6.8%` ✕ | 🏆 **`0.1%`（`InferenceModel` 85% KV 水位自动拦截批处理）** |
| **月度达标 GPU 集群总成本 (`TCO`)** | `$148,000 / 月` | `$126,000 / 月` | `$92,000 / 月` | **`$54,000 / 月` 🏆（节省 63.5% 算力成本）** |

---

## ⑤ 🧪 定量算例、交互式容量计算器与 7-Card 4:5 视觉报告

### 5.1 基准算例复核（64×H100 80GB / 16 副本 / 1,200 RPS / 48 个 LoRA 适配器）

- **交互式定容计算器（CSP 自包含 HTML）**：👉 [打开 Kubernetes Gateway API Inference Extension 与 EPP 定容计算器](../reports/enterprise-k8s-gateway-api-inference-extension-llm-d-kv-epp-routing-sizer.html)
- **7-Card 4:5 视觉评测卡片报告**：👉 [打开 7-Card 视觉评测报告](../reports/enterprise-k8s-gateway-api-inference-extension-llm-d-kv-epp-routing-selection-cards.html)
- **可复核确定性脚本**：
  ```bash
  python experiments/enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection/run.py --check
  python -m unittest tests/test_enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection.py
  ```

### 5.2 核心量化对比图（Prefix/LoRA 缓存命中率 % vs. HBM KV 抢占驱逐率 %）

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: K8s Round-Robin", "metric": "Prefix & LoRA Hit Rate (%)", "pct": 16.5 },
      { "tier": "Tier 1: K8s Round-Robin", "metric": "KV Preemption Drop (%)", "pct": 14.8 },
      { "tier": "Tier 2: HTTP Least-Conn", "metric": "Prefix & LoRA Hit Rate (%)", "pct": 21.0 },
      { "tier": "Tier 2: HTTP Least-Conn", "metric": "KV Preemption Drop (%)", "pct": 8.6 },
      { "tier": "Tier 3: Client Sticky Hash", "metric": "Prefix & LoRA Hit Rate (%)", "pct": 64.0 },
      { "tier": "Tier 3: Client Sticky Hash", "metric": "KV Preemption Drop (%)", "pct": 5.4 },
      { "tier": "Tier 4: Gateway API EPP + llm-d", "metric": "Prefix & LoRA Hit Rate (%)", "pct": 89.4 },
      { "tier": "Tier 4: Gateway API EPP + llm-d", "metric": "KV Preemption Drop (%)", "pct": 0.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "metric": "Category",
    "pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Grouped Bar Chart",
    "title": "Tier 4 boosts prefix/LoRA hit rate to 89.4% and eliminates HBM KV preemption drops",
    "subtitle": "Comparison across a 64-GPU H100 Kubernetes cluster at 1,200 RPS",
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

- [ ] **严禁在 `vLLM / SGLang / TensorRT-LLM` 推理 Pod 前直接使用默认 Kubernetes `Service` 轮询或 HTTP 最小连接数（Least-Connections）切流**：必须部署支持 `Gateway API Inference Extension`（`InferencePool` + `EPP`）的推理网关（如 Envoy Gateway / Kgateway / Istio + `llm-d`）。
- [ ] **端点选择器（`EPP`）必须融合三大实时信号（`Pod KV Cache 占用率`、`等待队列深度`、`前缀块哈希与 LoRA 驻留位图`）**：当首选缓存命中 Pod 队列深度超阈值时自动向温备副本平滑溢出。
- [ ] **在 `EPP` 内部开启“在途未落盘请求 Token 本地预扣账本（In-Flight Token Accounting）”**：消除 Prometheus/gRPC 指标采集间隔（`100–250ms`）内的高并发羊群盲目派发。
- [ ] **通过 `InferenceModel` CRD 强制配置三级业务优先级（`Critical`、`Standard`、`Sheddable`）**：当集群聚合 KV 显存占用突破 `85%` 时自动拒绝/延后 `Sheddable` 离线批处理请求。
- [ ] **将 KEDA 自动扩缩容（HPA）触发器从“CPU 利用率”切换为“集群平均 `kv_cache_usage_perc` 与 `num_requests_waiting` 队列等待时长”**：并配合 Kubernetes `EndpointSlice` 优雅终止信号（`terminating`）实现滚动升级零丢包。
