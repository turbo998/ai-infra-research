# Enterprise Cloud-Native Kubernetes Gateway API Inference Extension (`InferencePool` + `EPP`), `llm-d` KV/LoRA-Aware Routing & Criticality Load-Shedding Architecture Selection Guide: Standard K8s `Service` Round-Robin vs. HTTP Least-Connections Ingress vs. Client-Side Sticky Hashing vs. Envoy `ext_proc` Endpoint Picker (`EPP`) + `KV Block-Hash / LoRA` Composite Scorer + 85% Watermark Criticality Shedder

> **Language / 语言**: **English (Current)** | [中文](./enterprise-k8s-gateway-api-inference-extension-llm-d-kv-epp-routing-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Kubernetes Platform Engineering Leads (`AKS / EKS / GKE / OpenShift`), and Enterprise LLM Serving (`vLLM / SGLang / llm-d`) Architects.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection/results.json`](../../experiments/enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection/results.json).

![Enterprise Kubernetes Gateway API Inference Extension & llm-d KV/LoRA Routing Summary](../../assets/svg/enterprise-k8s-gateway-api-inference-extension-llm-d-kv-epp-routing-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating a **64-GPU NVIDIA H100 (80GB) Kubernetes inference cluster with 16 `vLLM / SGLang` replicas serving `1,200 RPS` of mixed-length enterprise traffic and `48` departmental fine-tuned LoRA adapters** exposes three fatal flaws in traditional microservice ingress controllers (`kube-proxy`, Nginx, or standard Envoy Round-Robin / Least-Connections):
1. **HTTP Least-Connections Is Blind to `100x` Token-Length Asymmetry and GPU HBM KV Saturation**: Pod A holds only `2` active TCP connections, but both are `64K-token` RAG requests consuming **`94%` of HBM KV cache**; Pod B holds `8` active `256-token` chat requests consuming only **`12%` of HBM KV cache**. A standard Least-Connections load balancer (Tier 2) routes a new `32K` prompt to Pod A (`2 < 8`), triggering immediate **PagedAttention KV cache preemption/recomputation (`8.6%–14.8%` drop rate and `3,120–4,480 ms` P99 TTFT)**!
2. **Round-Robin Destroys Prefix Cache & LoRA Residency (`16.5%` Hit Rate), While Static Client Sticky Hashing (Tier 3) Suffers `1,450 ms` Hot-Pod Queue Stalls Under Tenant Bursts**.
3. **Unprioritized Admission Lets Background Batch Jobs Starve Live Customer Copilots (`22.4%` Critical SLO Breach Rate)**.

Evaluated across **64 H100 GPUs, 16 `vLLM` pods, 1,200 RPS, and 48 LoRA adapters**, the four Kubernetes inference routing tiers compare as follows:

| Architectural Tier | Core Stack Combination | P99 TTFT Latency | Prefix & LoRA Hit Rate | KV Preemption / OOM Drop | Critical-Tier SLO Breach | Monthly GPU Cluster TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Standard K8s `Service` Round-Robin** | Default `kube-proxy` / L7 Round-Robin across `vLLM` pods | `4,480 ms` ✕ | `16.5%` ✕ *(8x prefill waste)* | `14.8%` ✕ *(KV thrashing)* | `22.4%` ✕ | `$148,000 / mo` ✕ | ⛔ Treats stateful KV engines like stateless web apps |
| **🟨 Tier 2: HTTP Least-Connections + CPU HPA** | Routing by active HTTP stream count + CPU/RAM HPA | `3,120 ms` ✕ | `21.0%` ✕ | `8.6%` ✕ *(64K hits full pod)* | `15.2%` ✕ | `$126,000 / mo` ✕ | ⛔ TCP stream count ignores prompt tokens & HBM KV |
| **🟦 Tier 3: Client Consistent-Hash Sticky Ring** | Static tenant/session hash pinning to fixed pods | `1,450 ms` | `64.0%` | `5.4%` ✕ *(hot-pod overload)* | `6.8%` ✕ | `$92,000 / mo` ✕ | ⚠️ Hot-spot queue stalls when large tenants burst |
| **🟩 Tier 4: K8s `Gateway API Inference Extension` (`EPP`) + `llm-d` Scorer** | **`InferencePool` + Envoy `ext_proc` Endpoint Picker (`EPP`) KV/Queue/Prefix/LoRA Scorer + `InferenceModel` 85%-KV Criticality Shedder** | **`340 ms` 🏆** *(-92.4%)* | **`89.4%` 🏆** *(+72.9 pts)* | **`0.0%` 🏆** *(Zero KV swap)* | **`0.1%` 🏆** *(SLO protected)* | **`$54,000 / mo` 🏆** *(-63.5%)* | 🌟 **Recommended Cloud-Native K8s Standard** |

> 💡 **Golden Rule for Cloud-Native Kubernetes AI Infrastructure Architects**: **"Never put a stateless Round-Robin or HTTP Least-Connections load balancer in front of stateful `vLLM / SGLang` pods—deploy the Kubernetes `Gateway API Inference Extension` (`InferencePool` + `InferenceModel` + Envoy `ext_proc` Endpoint Picker `EPP`) to score candidate pods on real-time `HBM KV-cache utilization + waiting queue depth + prefix block-hash / LoRA residency affinity`, while shedding `Sheddable` batch traffic whenever pool KV utilization exceeds `85%`."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Kubernetes LLM Routing Pain Points)

### Q1: Why do HTTP Least-Connections and CPU HPA fail catastrophically on Kubernetes LLM clusters?
- **First-Principles Reality**:
  - Unlike stateless REST microservices where every request costs roughly equal CPU, a single `64K-token` prompt consumes **`128x` more GPU prefill FLOPs and HBM KV blocks** than a `512-token` chat turn.
  - Worse, GPU-bound `vLLM` pods often run at `<20%` host CPU utilization even when HBM KV cache is `95%` full. Tier 4 replaces connection-count routing and CPU HPA with real-time pod telemetry (`kv_cache_usage_perc` and `num_requests_waiting`) consumed directly by the **`EPP` (Endpoint Picker)** and KEDA.

### Q2: How does the `EPP` Composite Scorer achieve an `89.4%` Prefix/LoRA cache hit rate without hot-pod queueing?
- **Multi-Signal Weighted Scoring + Local In-Flight Token Accounting**:
  - Before Envoy forwards a request, it invokes the **`EPP`** via `ext_proc` gRPC to score every pod $p$ in the `InferencePool`:
    $$\text{Score}(p, r) = w_{\text{prefix}} \cdot \text{PrefixBlockMatch}(p, r) + w_{\text{lora}} \cdot \mathbb{I}(\text{LoRA}_r \in p) - w_{\text{kv}} \cdot \left(U_{\text{KV}}(p) + \frac{\text{InFlightTokens}(p)}{C_{\text{KV}}(p)}\right) - w_{\text{q}} \cdot Q_{\text{wait}}(p)$$
  - By maintaining a local **In-Flight Dispatched Token Ledger** (`InFlightTokens(p)`) between `250ms` metric scrapes and penalizing queue depth $Q_{\text{wait}}(p)$, the `EPP` pins multi-turn requests to warm prefix/LoRA pods under normal load (`89.4%` hit rate) and smoothly spills over to secondary replicas during bursts (`0.0%` KV preemption drops).

### Q3: How does `InferenceModel` Criticality Load Shedding prevent batch workloads from breaking live Copilot SLAs?
- **3-Band Priority Admission (`Critical` / `Standard` / `Sheddable`) at the `85%` KV Watermark**:
  - The `InferenceModel` CRD maps logical workload endpoints (`copilot-live`, `internal-search`, `batch-summarizer`) to explicit criticality bands on the shared `InferencePool`.
  - Once aggregate pool KV utilization reaches **`85%`**, the `EPP` immediately sheds `Sheddable` batch requests (`HTTP 429` with retry-after backoff), reserving the remaining `15%` HBM KV headroom for `Critical` interactive traffic (`0.1%` SLO breach rate vs. `22.4%` in Tier 1).

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + `InferencePool / EPP` Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Kubernetes LLM Inference Cluster Ingress<br/>(64xH100 + 16 vLLM/SGLang Pods + 1,200 RPS + 48 LoRA Adapters)"] --> B{"Uses default K8s Service Round-Robin<br/>or HTTP Least-Connections ingress?"}
    B -- "Yes (Stateless L4/L7 routing)" --> C["⛔ KV Preemption Thrashing & Cache Misses (Tier 1 / Tier 2)!<br/>4,480ms P99 TTFT, 16.5% prefix hit rate & 14.8% HBM KV preemption drops"]
    B -- "Uses affinity routing" --> D{"Uses K8s Gateway API Inference Extension (EPP)<br/>with KV/Queue/Prefix/LoRA composite scoring & criticality shedding?"}
    D -- "No (Static client sticky hash ring)" --> E["⚠️ Hot-Pod Queue Stalls Under Burst (Tier 3)<br/>1,450ms P99 TTFT when bursty tenants overload a single pinned pod"]
    D -- "Yes (EPP Composite Scorer + 85% KV Gate)" --> F["🟩 Tier 4: K8s Gateway API Inference Extension + llm-d EPP 🏆<br/>(340 ms P99 TTFT | 89.4% Prefix/LoRA Hit | 0.0% KV Drop | -63.5% GPU TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 `Gateway API Inference Extension` (`InferencePool` + `InferenceModel` + `EPP`) Topology

```mermaid
flowchart LR
    subgraph CRDs["1. Workload Criticality CRDs (InferenceModel)"]
        M1["🟢 Critical: Live Customer Copilot"]
        M2["🔵 Standard: Internal Knowledge Search"]
        M3["🟡 Sheddable: Batch Doc Summarizer"]
    end

    subgraph Gateway["2. Envoy Gateway + ext_proc Endpoint Picker (EPP)"]
        M1 & M2 & M3 --> Envoy["🌐 Envoy AI Gateway"]
        Envoy <-->|gRPC ext_proc| EPP["🧠 Endpoint Picker (EPP)<br/>1. Sheds Sheddable at >=85% KV<br/>2. Score: PrefixHash + LoRA - KV% - Queue<br/>3. Local In-Flight Token Accounting"]
    end

    subgraph Pool["3. InferencePool (16x vLLM / SGLang Replicas)"]
        EPP -->|Bypasses saturated pod| PodA["Pod A (KV: 94% -> Skipped)"]
        EPP -->|Routes to warm Prefix+LoRA| PodB["🏆 Pod B (KV: 28%, Warm LoRA)"]
        PodA & PodB -.->|Real-time KV & Queue Metrics| EPP
    end
```

### 3.3 End-to-End Sequence: Routing a `32K` Prompt Around a Saturated Pod to a Warm Prefix/LoRA Replica

```mermaid
sequenceDiagram
    autonumber
    participant App as Enterprise Legal Copilot Client
    participant Envoy as Envoy Gateway (Gateway API)
    participant EPP as Endpoint Picker (EPP ext_proc)
    participant PodA as vLLM Pod A (2x 64K reqs, KV=94%)
    participant PodB as vLLM Pod B (8x 256-tok reqs, KV=12%, Warm LoRA)

    App->>Envoy: POST /v1/chat/completions (model="lora-legal-v3", 32K prompt w/ 24K shared contract prefix)
    Envoy->>EPP: ext_proc gRPC callout (criticality="Critical", prefix_block_hashes, target_lora)
    EPP->>EPP: 1. Exclude Pod A (KV=94% > 85% threshold -> prevents HBM KV eviction swap!)
    EPP->>EPP: 2. Score Pod B: +0.45 (24K prefix match) + 0.30 (lora-legal-v3 resident) - 0.06 (12% KV) = Best Pod!
    EPP->>EPP: 3. Increment local in-flight token ledger on Pod B (+8K uncached tokens)
    EPP-->>Envoy: Return header x-gateway-destination-endpoint: Pod B (10.244.3.18:8000)
    Envoy->>PodB: Forward request -> reuses 24K prefix KV & warm LoRA in HBM (340 ms P99 TTFT!)
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: K8s `Service` Round-Robin | 🟨 Tier 2: HTTP Least-Conn + CPU HPA | 🟦 Tier 3: Client Sticky Hash Ring | 🟩 Tier 4: `Gateway API Inference Extension` (`EPP`) |
|---|---|---|---|---|
| **P99 TTFT Latency** | `4,480 ms` ✕ | `3,120 ms` ✕ | `1,450 ms` | 🏆 **`340 ms` (`-92.4%` vs. baseline)** |
| **Prefix Cache & LoRA Hit Rate** | `16.5%` ✕ (random pod scatter) | `21.0%` ✕ | `64.0%` (no queue overflow) | 🏆 **`89.4%` (KV block-hash + LoRA bitmap + overflow)** |
| **HBM KV Preemption / OOM Drop** | `14.8%` ✕ (frequent KV eviction) | `8.6%` ✕ (64K sent to full pod) | `5.4%` ✕ (hot-pod saturation) | 🏆 **`0.0%` (real-time KV filter + in-flight ledger)** |
| **Critical-Tier SLO Breach Rate** | `22.4%` ✕ (batch starvation) | `15.2%` ✕ | `6.8%` ✕ | 🏆 **`0.1%` (`InferenceModel` 85% KV criticality gate)** |
| **Monthly GPU Cluster Fleet TCO** | `$148,000 / mo` | `$126,000 / mo` | `$92,000 / mo` | **`$54,000 / mo` 🏆 (-63.5%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise K8s Gateway API Inference Extension & llm-d EPP Sizer](../reports/enterprise-k8s-gateway-api-inference-extension-llm-d-kv-epp-routing-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-k8s-gateway-api-inference-extension-llm-d-kv-epp-routing-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection/run.py --check
  python -m unittest tests/test_enterprise_k8s_gateway_api_inference_extension_llm_d_kv_epp_routing_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Replace Stateless `kube-proxy` Round-Robin and HTTP Least-Connections Ingress in Front of `vLLM / SGLang` Fleets**: Deploy a Gateway API controller supporting the official `Gateway API Inference Extension` (`InferencePool` + `InferenceModel` + Envoy `ext_proc` `EPP`).
- [ ] **Configure the Endpoint Picker (`EPP`) to Score Pods on `HBM KV Utilization + Waiting Queue Depth + Prefix Block-Hash / LoRA Residency`**: Spill over gracefully to warm secondary replicas when primary pod queue depth exceeds threshold.
- [ ] **Enable Local In-Flight Dispatched Token Accounting Inside the `EPP`**: Prevent herd routing during the `100–250ms` window between pod metric scrapes.
- [ ] **Enforce 3-Band Workload Criticality (`Critical`, `Standard`, `Sheddable`) via `InferenceModel` CRDs**: Shed `Sheddable` batch requests when pool-wide KV cache utilization exceeds `85%`.
- [ ] **Drive KEDA Horizontal Pod Autoscaling from `kv_cache_usage_perc` and `num_requests_waiting` Rather Than Host CPU Utilization**: Respect `EndpointSlice` `terminating` conditions for zero-drop rolling upgrades.
