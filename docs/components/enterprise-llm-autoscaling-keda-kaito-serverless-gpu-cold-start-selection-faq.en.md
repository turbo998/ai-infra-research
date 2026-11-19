# Enterprise LLM Inference Autoscaling, GPU Cold-Start Acceleration & Burst Spillover Architecture Selection Guide: Static 24x7 Peak GPU Fleet vs. Standard Kubernetes `GPU_UTIL HPA` vs. `KEDA` Queue Scaling vs. `KEDA` KV-Cache Metrics + `KAITO/NVMe` 36-Second Fast Weight Streaming + AI Gateway Cloud `PAYGO` Spillover

> **Language / 语言**: **English (Current)** | [中文](./enterprise-llm-autoscaling-keda-kaito-serverless-gpu-cold-start-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and FinOps elasticity guide for Cloud & AI Solution Architects, Kubernetes (`AKS / EKS / GKE`) Platform Leads, and AI Infrastructure FinOps Engineers.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection/results.json`](../../experiments/enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection/results.json).

![Enterprise LLM Autoscaling, GPU Cold-Start & Burst Spillover Summary](../../assets/svg/enterprise-llm-autoscaling-keda-kaito-serverless-gpu-cold-start-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Enterprise AI Copilot, customer support, and coding assistant workloads exhibit a pronounced **diurnal peak-to-trough ratio of `8.5 : 1`**—requiring **32x H100 GPUs** during peak business hours (09:00–17:00) while dropping by nearly 90% overnight and on weekends. Architects designing Kubernetes (`AKS / EKS / GKE`) LLM inference autoscaling routinely hit three production traps:
1. **Static 24x7 Peak Provisioning (Tier 1, `26.5%` Average Fleet GPU Utilization)**: Keeping 32x H100 GPUs running 24x7 wastes 73.5% of GPU hours overnight and on weekends, burning **`$198,000/month`**;
2. **Standard Kubernetes HPA on `CPU` or `DCGM_FI_DEV_GPU_UTIL` (Tier 2, `34.2%` Surge Timeout Rate, `18.40s` P99 TTFT)**: Autoregressive LLM decoding is **KV-cache memory-capacity and bandwidth bound**, not compute bound. Under long-context load, `vLLM` KV-cache hits **`99%` full** and 80 requests queue up while GPU compute utilization (`GPU_UTIL`) sits at just **`52%`**—so standard HPA never triggers scale-out!
3. **`KEDA` Queue Scaling with Slow Network Weight Downloads and No Cloud Spillover (Tier 3, `270s` Cold Start, `11.8%` Surge Drop)**: Even with the right KEDA metric, pulling a 15 GB container image and 72 GB of 70B FP8 weights over HTTP takes `270s–680s` (4.5 to 11.3 minutes), leaving callers timing out during the boot window.

Evaluated across a **32x H100 peak diurnal workload, 70B FP8 model (`72 GB` weights), `8.5:1` peak-to-trough ratio, and a `1.50s` P99 TTFT SLA**, the four autoscaling architectures compare as follows:

| Architectural Tier | Core Stack Combination | 70B Replica Ready Latency | 5x Surge Drop / Timeout | 24x7 Fleet GPU Utilization | Surge P99 `TTFT` | Monthly Compute TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Static 24x7 Peak Fleet (`32x H100`)** | Fixed 32x H100 running 24x7 with no autoscaling | `0.0s` *(always warm)* | `0.5%` | `26.5%` ✕ *(73.5% idle waste)* | `1.15s` | `$198,000 / mo` ✕ | ⛔ Massive FinOps waste overnight and on weekends |
| **🟨 Tier 2: Standard K8s HPA (`GPU_UTIL`)** | Scales on `DCGM_FI_DEV_GPU_UTIL > 80%` + standard HTTP pull | `680.0s (11.3 min)` ✕ | `34.2%` ✕ | `48.0%` | `18.40s` ✕ *(KV full, HPA blind)* | `$134,000 / mo` ✕ | ⛔ #1 LLM autoscaling anti-pattern |
| **🟦 Tier 3: `KEDA` KV-Cache Only (Slow Pull)** | Scales on `vllm:gpu_cache_usage_perc` + 4.5-min weight load | `270.0s (4.5 min)` ✕ | `11.8%` ✕ *(boot-window drops)* | `64.5%` | `6.80s` ✕ | `$102,000 / mo` | ⚠️ Right trigger metric, but too slow on cold start |
| **🟩 Tier 4: `KEDA` KV-Cache + `KAITO/NVMe` 36s Stream + Cloud Spillover** | **`KEDA (KV-Cache + Queue)` + OCI Streaming / NVMe Direct Tensor Load (`36s`) + AI Gateway Cloud PAYGO Burst Spillover** | **`36.0s` 🏆** *(94.7% faster)* | **`0.0%` 🏆** *(Cloud bridges 36s)* | **`82.4%` 🏆** | **`0.92s` 🏆** | **`$78,000 / mo` 🏆** *(-60.6%)* | 🌟 **Recommended Enterprise Elastic LLM Architecture** |

> 💡 **Golden Rule for Enterprise LLM Autoscaling Architects**: **"Never scale LLM serving pods on `GPU_UTIL`. Combine three mechanisms: (1) trigger `KEDA` scale-out on leading indicators (`vllm:gpu_cache_usage_perc > 75%` or `num_requests_waiting > 5`); (2) slash 70B replica cold start from 11 minutes to `36 seconds` using OCI Artifact Streaming and local NVMe parallel tensor streaming (`KAITO / Run:ai Model Streamer`); and (3) bridge the 36-second boot window by having the AI Gateway spill excess queue traffic over to Cloud Serverless PAYGO."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 LLM Autoscaling & Cold-Start Pain Points)

### Q1: Why does standard Kubernetes HPA on `DCGM_FI_DEV_GPU_UTIL` fail to scale out when `vLLM / SGLang` is completely saturated?
- **First-Principles Reality**:
  - LLM token generation is **memory-bandwidth and KV-cache capacity bound**. Once `vllm:gpu_cache_usage_perc` reaches `98%–100%`, `vLLM` cannot admit new sequences into the running batch and queues them in `vllm:num_requests_waiting`, even while GPU compute utilization (`DCGM_FI_DEV_GPU_UTIL`) hovers around `50%`.
  - Tier 4 uses **KEDA Prometheus Triggers** on `vllm:gpu_cache_usage_perc > 0.75` and `vllm:num_requests_waiting > 5`, plus a `KEDA Cron` pre-warming schedule at 08:45 AM on weekdays.

### Q2: How do we compress a 70B FP8 (`72 GB`) pod cold start from `680 seconds` down to `36 seconds`?
- **Three-Stage Cold-Start Pipeline**:
  1. **Container Image OCI Streaming (`4.0s`)**: AKS Artifact Streaming / AWS SOCI mounts the 15 GB vLLM/CUDA container without waiting for a full image download;
  2. **Parallel Direct-to-GPU Tensor Streaming (`24.0s`)**: Pre-staged weights on local NVMe RAID-0 / `KAITO` immutable disk images are streamed directly into H100 HBM3 via `Run:ai Model Streamer` / `safetensors` multi-threaded DMA at `12+ GB/s`;
  3. **Persistent CUDA Graph Cache (`8.0s`)**: Mounting pre-compiled `torch.compile` and CUDA Graph artifacts avoids multi-minute kernel compilation on startup.

### Q3: How do we prevent request drops during the 36-second pod boot window and during evening scale-in?
- **AI Gateway Cloud PAYGO Burst Spillover + Graceful Connection Draining**:
  - During the `0s–36s` window while KEDA brings new local GPU pods online, the AI Gateway overflows queued requests above threshold to **Azure OpenAI / AWS Bedrock Serverless PAYGO** (`0.0%` dropped requests, `0.92s` P99 TTFT).
  - On scale-in, `terminationGracePeriodSeconds: 180` and a `PreStop` hook wait until `vllm:num_requests_running == 0` before terminating the pod (`0.0%` mid-stream disconnects).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + KEDA/KAITO/Spillover Topology + Sequence Flow)

### 3.1 Enterprise LLM Inference Autoscaling Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Kubernetes LLM Inference Autoscaling Selection<br/>(32x H100 Peak + 70B FP8 Model + 8.5:1 Diurnal Ratio)"] --> B{"Release idle GPUs overnight and on weekends?"}
    B -- "No (Static 24x7 32x H100 Fleet)" --> C["⛔ 73.5% Idle GPU Waste (Tier 1)<br/>$198,000/month compute bill for 26.5% average utilization"]
    B -- "Yes (Dynamic Autoscaling)" --> D{"Which scaling metrics and weight loading pipeline?"}
    D -- "CPU / GPU_UTIL HPA + standard HTTP pull" --> E["⛔ HPA Blindness & 11-Min Cold Start (Tier 2)<br/>KV-cache hits 99% at 52% GPU_UTIL -> 34.2% surge timeouts"]
    D -- "KEDA KV-Cache/Queue + 36s NVMe Stream + Cloud Spillover" --> F["🟩 Tier 4: KEDA KV-Cache + KAITO 36s + Cloud Spillover 🏆<br/>(36s Ready | 0.0% Surge Timeout | 82.4% GPU Util | -60.6% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `KEDA` KV-Cache + `KAITO/NVMe` 36-Second Loader + Cloud Spillover Reference Architecture

```mermaid
flowchart LR
    subgraph Gateway["1. AI Gateway Burst Spillover Layer"]
        Req["Morning 5x Traffic Surge"] --> AIGW["🌐 AI Gateway (Envoy / APIM)<br/>Routes to local H100 pool; overflows<br/>to Cloud PAYGO when queue > 5"]
    end

    subgraph Cluster["2. KEDA + KAITO 36s Elastic GPU Cluster"]
        AIGW -->|Primary Traffic| VLLM["🚀 Self-Hosted vLLM / SGLang Pool<br/>Exports gpu_cache_usage_perc & queue metrics"]
        VLLM -->|KV > 75% or Queue > 5| KEDA["📈 KEDA Autoscaler + Cron Warmup<br/>Triggers instant scale-out & graceful drain"]
        KEDA --> FastBoot["⚡ 36s Cold-Start Pipeline<br/>OCI Streaming (4s) + NVMe Tensor Stream (24s)"]
        AIGW -.->|Spillover during 0s..36s boot gap| Cloud["☁️ Azure OpenAI / Bedrock PAYGO<br/>0.0% Surge Timeout Bridge"]
    end
```

### 3.3 End-to-End Sequence: Handling a 5x Monday 09:00 AM Surge with Zero Timeouts

```mermaid
sequenceDiagram
    autonumber
    participant User as Enterprise Users (5x Morning Surge)
    participant GW as AI Gateway (Spillover Router)
    participant VLLM as Local vLLM H100 Pool
    participant KEDA as KEDA + KAITO Fast Loader
    participant Cloud as Cloud Serverless PAYGO API

    User->>GW: t=0s: Monday 09:00 AM traffic surges 5x
    GW->>VLLM: Route to active replicas; `gpu_cache_usage_perc` crosses 75% & waiting queue > 5
    VLLM->>KEDA: t=2s: KEDA Prometheus trigger fires scale-out for 8 additional H100 replicas
    GW->>Cloud: t=2s..38s (36s boot window): Gateway spills excess queued calls to Cloud PAYGO (0.0% drops, 0.92s P99 TTFT!)
    KEDA->>VLLM: t=38s: New replicas finish OCI mount + 24s NVMe direct tensor streaming into HBM3 -> Ready!
    GW->>VLLM: t>38s: 100% traffic served by self-hosted H100 pool at 82.4% 24x7 GPU utilization!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Static 24x7 (`32x H100`) | 🟨 Tier 2: Standard `GPU_UTIL HPA` | 🟦 Tier 3: `KEDA` Only (Slow Pull) | 🟩 Tier 4: `KEDA` + `36s Loader` + `Spillover` |
|---|---|---|---|---|
| **Autoscaling Metric Accuracy** | None (static 32 GPUs) | ✕ Broken (KV full while `GPU_UTIL` is 52%) | High (KV-cache + queue depth) | 🏆 **Optimal (KV-Cache + Queue + Weekday Cron Warmup)** |
| **70B Replica Ready Latency** | `0.0s` (always burning power) | `680.0s` (11.3 min) ✕ | `270.0s` (4.5 min) ✕ | 🏆 **`36.0s` (OCI Streaming + NVMe Direct Tensor Stream)** |
| **5x Surge Timeout / Drop Rate** | `0.5%` | `34.2%` ✕ | `11.8%` ✕ (drops during 270s boot) | 🏆 **`0.0%` (AI Gateway spills over to Cloud PAYGO for 36s)** |
| **24x7 Fleet GPU Utilization** | `26.5%` ✕ | `48.0%` | `64.5%` | 🏆 **`82.4%` (scales down cleanly overnight/weekends)** |
| **Monthly Inference Compute TCO** | `$198,000 / mo` | `$134,000 / mo` | `$102,000 / mo` | **`$78,000 / mo` 🏆 (-60.6%, saves $120K/mo)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise LLM Inference Autoscaling, Cold-Start & Burst Spillover Sizer](../reports/enterprise-llm-autoscaling-keda-kaito-serverless-gpu-cold-start-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-llm-autoscaling-keda-kaito-serverless-gpu-cold-start-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection/run.py --check
  python -m unittest tests/test_enterprise_llm_autoscaling_keda_kaito_serverless_gpu_cold_start_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Replace `CPU` and `DCGM_FI_DEV_GPU_UTIL` HPA Rules with `KEDA` Prometheus Triggers on `vllm:gpu_cache_usage_perc` (`> 75%`) and `vllm:num_requests_waiting` (`> 5`)**.
- [ ] **Enable Container Artifact Streaming (`AKS Artifact Streaming / AWS SOCI`) and Local NVMe Parallel Tensor Loading (`KAITO / Run:ai Model Streamer`)**: Persist compiled CUDA Graph caches so 70B FP8 replicas reach `Ready` in `<= 40 seconds`.
- [ ] **Configure AI Gateway Burst Spillover to Cloud Serverless `PAYGO` (`Azure OpenAI / AWS Bedrock`)**: Absorb excess queue spikes during the 36-second scale-out window with zero HTTP 503 errors.
- [ ] **Schedule Weekday Morning `KEDA Cron` Pre-Warming 15 Minutes Before Business-Hour Ramp**: Avoid cold-starting from minimum night capacity at 09:00 AM.
- [ ] **Enforce Graceful Connection Draining (`terminationGracePeriodSeconds: 180` + `PreStop` Check for `num_requests_running == 0`)**: Prevent mid-stream disconnects during evening scale-in.
