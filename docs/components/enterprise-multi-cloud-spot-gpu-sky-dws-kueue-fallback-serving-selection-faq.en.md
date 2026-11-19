# Enterprise Multi-Cloud Spot / Preemptible GPU Orchestration, Atomic Gang Provisioning (`Kueue ProvisioningRequest / GCP DWS / AWS Capacity Blocks`), `SkyPilot` Cross-Pool Broker & `30s IMDS` Drain-Aware Serving Architecture Selection Guide: Single-AZ On-Demand Autoscaler vs. Raw Single-Zone Spot vs. Single-Cloud Multi-AZ Spot vs. Atomic `Kueue DWS` + `30% Base / 70% 4-Pool Spot` + `30s IMDS` Zero-Drop Drain

> **Language / 语言**: **English (Current)** | [中文](./enterprise-multi-cloud-spot-gpu-sky-dws-kueue-fallback-serving-selection-faq.md)  
> **Scope**: End-to-end production reference architecture and TCO selection guide for Cloud & AI Solution Architects, Multi-Cloud Infrastructure & FinOps Leads, and Kubernetes (`EKS / GKE / AKS`) AI Platform Engineers.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection/results.json`](../../experiments/enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection/results.json).

![Enterprise Multi-Cloud Spot GPU, Kueue DWS & SkyPilot Fallback Serving Summary](../../assets/svg/enterprise-multi-cloud-spot-gpu-sky-dws-kueue-fallback-serving-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

Operating a **96-GPU (H100-equivalent) enterprise cloud fleet** for elastic LLM serving and distributed post-training exposes three costly cloud infrastructure failure modes:
1. **Standard Kubernetes Cluster Autoscaler Provisions Multi-Node Gang Jobs Node-by-Node (`22.0%` Partial-Allocation Idle Billing Waste)**: When a 16-GPU job requests `2×8 H100` nodes during a regional capacity crunch where only 1 node is in stock, standard autoscalers boot Node #1 immediately (`$32/hr`) and wait 4 hours for Node #2 (`InsufficientInstanceCapacity`)—burning budget while the distributed job cannot even initialize `NCCL`!
2. **Raw Spot GPUs Without `IMDS 30s–120s` Preemption Draining Drop `100.0%` of In-Flight SSE Streams on `SIGKILL` and Suffer `34.0%` Stockouts During Single-AZ Reclamation Sweeps**.
3. **Naive Cross-Cloud Failover Pulls 140GB Model Weights Over Public WAN Links, Incurring Thousands of Dollars in Egress Fees and 7-Minute Cold Starts**.

Evaluated across a **96-GPU enterprise fleet spanning 4 cloud regions and SKU pools**, the four cloud GPU procurement architectures compare as follows:

| Architectural Tier | Core Stack Combination | GPU Stockout Failure Rate | Preemption Dropped Requests | Partial-Gang Idle Waste | Failover Recovery Time | Monthly Cloud GPU TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Single-AZ On-Demand + Standard CA** | 100% On-Demand locked to 1 AZ + node-by-node Cluster Autoscaler | `28.4%` ✕ *(frequent AZ stockouts)* | `18.5%` ✕ *(abrupt scale-in kills)* | `22.0%` ✕ *(boots 1 of 2 gang nodes)* | `420.0 s` ✕ | `$215,000 / mo` ✕ | ⛔ Expensive, stockout-prone & wastes gang nodes |
| **🟨 Tier 2: Raw Single-Zone Spot (No Drain)** | Single-AZ Spot VMs without IMDS preemption draining or fallback | `34.0%` ✕ *(AZ-wide Spot sweeps)* | `100.0%` ✕ *(SIGKILL severs SSE)* | `19.5%` ✕ | `510.0 s` ✕ | `$142,000 / mo` ✕ | ⛔ Cheap hourly rate but constant HTTP 502s |
| **🟦 Tier 3: Single-Cloud Multi-AZ Spot Pool** | Multi-AZ Spot via Karpenter/NAP without atomic DWS or SKU fallback | `9.2%` | `4.8%` ✕ | `8.4%` ✕ | `135.0 s` | `$118,000 / mo` ✕ | ⚠️ Stalls when primary GPU SKU depletes region-wide |
| **🟩 Tier 4: Atomic `Kueue DWS` + `SkyPilot` 4-Pool Spot + `30s` Drain** | **All-or-Nothing `Kueue ProvisioningRequest` + `30% Base / 70% 4-Pool Spot` + `IMDS 30s` Envoy Drain + Zero-Egress `R2/P2P`** | **`0.1%` 🏆** *(-28.3 pts)* | **`0.0%` 🏆** *(Zero stream drop)* | **`0.0%` 🏆** *(Zero partial burn)* | **`14.0 s` 🏆** *(-96.7%)* | **`$79,000 / mo` 🏆** *(-63.3%)* | 🌟 **Recommended Cloud GPU Elasticity Standard** |

> 💡 **Golden Rule for Multi-Cloud AI Infrastructure Architects**: **"Never provision multi-node GPU workloads one VM at a time or run Spot inference without termination hooks—use Kubernetes `Kueue ProvisioningRequest` (`GCP DWS / AWS Capacity Blocks`) for atomic all-or-nothing gang provisioning (`0.0%` partial-allocation waste), diversify `70%` Spot capacity across `4` uncorrelated AZ/SKU equivalence pools (`H100 ↔ A100-80GB ↔ L40S`) anchored by a `30%` On-Demand base, and intercept cloud `IMDS 30s–120s` preemption notices at `t=0s` to cordon Envoy routing and gracefully drain active SSE streams."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Cloud GPU Procurement & Spot Pain Points)

### Q1: How does `Kueue ProvisioningRequest` (`GCP DWS / AWS Capacity Blocks`) eliminate the `22.0%` Partial-Gang Idle Billing Waste?
- **First-Principles Reality**:
  - When a distributed job needs $N$ nodes simultaneously, standard Kubernetes autoscalers request VMs individually. If the cloud zone only has $N-1$ nodes available, those $N-1$ GPU VMs boot and bill immediately while the job sits Pending in Kubernetes waiting for the final node.
  - Tier 4 wraps every multi-node job in a **`Kueue ProvisioningRequest`** CRD backed by cloud-native atomic reservation APIs (`GCP Dynamic Workload Scheduler DWS` or `AWS EC2 Capacity Blocks`). The cloud control plane holds VM boot until **100% of the requested gang capacity is guaranteed simultaneously** (`0.0%` partial-gang waste).

### Q2: How does `IMDS 30s–120s` Preemption Draining achieve `0.0%` dropped SSE requests on Spot GPUs?
- **Sub-Second Envoy Cordon + Active Stream Completion / Mid-Stream Handoff**:
  1. **Instant Cordon (`t = 0.1s`)**: A daemon polls the cloud instance metadata service (`169.254.169.254`) every `500ms`. Upon receiving a `30s` (GCP/Azure) or `120s` (AWS) preemption notice, it immediately removes the pod from the Envoy `InferencePool` so zero new requests arrive.
  2. **Graceful Drain (`t = 0.2s – 14.0s`)**: In-flight decode streams with `<300` remaining tokens finish naturally before `t = 15s`, while long-CoT streams execute a stateless **Mid-Stream Prefix Handoff** to a warm replica (`0.0%` client-visible drops).

### Q3: How does 4-Pool AZ/SKU Diversification survive an `80%` Single-AZ Spot Reclamation Storm without incurring cross-cloud egress fees?
- **Uncorrelated Fault Domains + Zero-Egress Weight Pre-Staging**:
  - Spreading Spot capacity across **4 uncorrelated pools** (2 Regions × 2 equivalent SKU families: `8×H100 FP8` $\leftrightarrow$ `8×A100-80GB` $\leftrightarrow$ `2×8 L40S`) reduces simultaneous stockout probability from $p = 34\%$ to $p^4 \approx 0.1\%$.
  - Pre-staging model weights in **zero-egress object storage (`Cloudflare R2` / regional buckets) + VPC-local `Dragonfly P2P`** ensures cross-region failover transfers only `4KB` JSON prompts rather than `140GB` weight blobs.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + Atomic DWS/Spot Topology + Sequence Flow)

### 3.1 Architecture Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Cloud GPU Procurement & Elastic Serving/Training<br/>(96-GPU H100-Equiv Fleet Across Multi-AZ / Multi-Region Pools)"] --> B{"Locked to single-AZ On-Demand autoscaling<br/>or raw un-drained single-zone Spot VMs?"}
    B -- "Yes (Single-AZ On-Demand or raw Spot)" --> C["⛔ Stockouts, Partial-Gang Burn & 100% Spot Drops (Tier 1 / Tier 2)!<br/>28.4%–34% stockouts, 22% partial-gang idle waste & severed SSE streams"]
    B -- "Uses multi-AZ Spot pools" --> D{"Uses atomic Kueue ProvisioningRequest (DWS), 4-pool<br/>AZ/SKU equivalence broker & IMDS 30s Envoy draining?"}
    D -- "No (Single-cloud multi-AZ Spot only)" --> E["⚠️ Regional SKU Depletion & Residual Drops (Tier 3)<br/>9.2% stockout rate when H100 depletes region-wide & 8.4% partial gang waste"]
    D -- "Yes (Kueue DWS + SkyPilot 4-Pool + 30s IMDS Drain)" --> F["🟩 Tier 4: Kueue DWS + SkyPilot 4-Pool Spot + 30s Drain 🏆<br/>(0.1% Stockout | 0.0% Preemption Drop | 0.0% Gang Waste | -63.3% TCO)"]
    style F fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Atomic `Kueue ProvisioningRequest` (`DWS`) + `SkyPilot` Multi-Pool + `30s` Drain Topology

```mermaid
flowchart LR
    subgraph Control["1. Atomic Gang & Multi-Pool Control Plane"]
        Job["Distributed Training / Elastic Serving"] --> Kueue["🧩 Kueue ProvisioningRequest<br/>GCP DWS / AWS Capacity Blocks<br/>All-or-Nothing Boot (0.0% Partial Burn)"]
        Job --> Sky["🌐 SkyPilot / Karpenter Broker<br/>30% On-Demand Base + 70% 4-Pool Spot<br/>(H100 ↔ A100-80G ↔ L40S Equivalence)"]
    end

    subgraph Drain["2. IMDS 30s–120s Graceful Preemption Drain"]
        IMDS["⚡ Cloud IMDS 30s Preempt Notice"] --> Hook["🛡️ Termination Handler (t=0.1s)"]
        Hook -->|1. Instant Envoy Cordon| Envoy["Envoy AI Gateway<br/>(Routes New Reqs + Mid-Stream Resume)"]
        Hook -->|2. 14s Active Stream Drain| Pod["vLLM Pod Exits Cleanly (0.0% Drop)"]
    end

    subgraph Storage["3. Zero-Egress Regional Weight Cache"]
        R2["📦 Cloudflare R2 (Zero Egress) / Regional S3<br/>+ Local Dragonfly P2P Streaming"] --> Pod
    end
```

### 3.3 End-to-End Sequence: Surviving a `30-Second` Spot Preemption Notice With Zero Dropped Streams

```mermaid
sequenceDiagram
    autonumber
    participant Cloud as Cloud Control Plane (IMDS / DWS)
    participant Node7 as Spot GPU Node #7 (IMDS Watcher)
    participant Envoy as Envoy AI Gateway (Global Router)
    participant PoolB as Equivalent Pool B (Region-B / A100-80G)

    Cloud->>Node7: t = 0.0s: Emit 30-second Spot preemption notice via IMDS (169.254.169.254)
    Node7->>Envoy: t = 0.1s: Mark pod Draining & remove Node #7 from Envoy InferencePool (0 new requests routed)
    Envoy->>PoolB: t = 0.2s: Shift incoming traffic to Pool B & trigger SkyPilot replacement Spot node boot
    Node7->>Node7: t = 0.2s–13.8s: Finish 18 active SSE decode streams; hand off 1 long-CoT stream prefix to Pool B
    PoolB-->>Envoy: t = 14.0s: Replacement Spot node streams weights from local Dragonfly P2P cache & joins pool!
    Cloud->>Node7: t = 30.0s: Cloud reclaims VM -> Node #7 has 0 active streams (0.0% HTTP 502 client errors!)
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Single-AZ On-Demand | 🟨 Tier 2: Raw Single-Zone Spot | 🟦 Tier 3: Multi-AZ Spot Only | 🟩 Tier 4: `Kueue DWS` + 4-Pool Spot + `30s` Drain |
|---|---|---|---|---|
| **GPU Stockout Failure Rate** | `28.4%` ✕ (single-AZ H100 scarcity) | `34.0%` ✕ (AZ reclamation sweeps) | `9.2%` | 🏆 **`0.1%` (4 uncorrelated AZ/SKU equivalence pools)** |
| **Preemption Dropped Requests** | `18.5%` ✕ (abrupt scale-in kills) | `100.0%` ✕ (`SIGKILL` severs SSE) | `4.8%` ✕ | 🏆 **`0.0%` (`IMDS t=0s` Envoy cordon + stream drain)** |
| **Partial-Gang Idle Billing Waste** | `22.0%` ✕ (boots 1 of 2 gang VMs) | `19.5%` ✕ | `8.4%` ✕ | 🏆 **`0.0%` (`Kueue ProvisioningRequest` all-or-nothing)** |
| **Failover & Replacement Time** | `420.0 s` ✕ | `510.0 s` ✕ | `135.0 s` | 🏆 **`14.0 s` (warm spillover + local P2P weight load)** |
| **Monthly Cloud GPU Fleet TCO** | `$215,000 / mo` | `$142,000 / mo` | `$118,000 / mo` | **`$79,000 / mo` 🏆 (-63.3%)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Multi-Cloud Spot GPU, Kueue DWS & SkyPilot Fallback Sizer](../reports/enterprise-multi-cloud-spot-gpu-sky-dws-kueue-fallback-serving-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-multi-cloud-spot-gpu-sky-dws-kueue-fallback-serving-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection/run.py --check
  python -m unittest tests/test_enterprise_multi_cloud_spot_gpu_sky_dws_kueue_fallback_serving_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Mandate Atomic All-or-Nothing `Kueue ProvisioningRequest` (`GCP DWS / AWS Capacity Blocks`) for All Multi-Node GPU Jobs**: Never boot and pay for partial multi-node clusters while waiting for scarce remaining nodes.
- [ ] **Anchor `70%` Elastic Spot Capacity with a `30%` On-Demand / Reserved Base Across `4+` Uncorrelated AZ & SKU Equivalence Pools**: Define tensor-parallel equivalence mappings (`8×H100` $\leftrightarrow$ `8×A100-80GB` $\leftrightarrow$ `2×8 L40S`) to survive regional single-SKU stockouts.
- [ ] **Wire Cloud `IMDS 30s–120s` Preemption Notices Directly Into Envoy Gateway Endpoint Draining at `t = 0.1s`**: Stop new request dispatch immediately and drain or hand off active SSE streams before `SIGKILL`.
- [ ] **Trigger Two-Phase Host-Pinned DRAM Async Checkpoints on Spot Training Nodes Upon `IMDS` Preemption Signals**: Flush distributed training state in `<5 seconds` well within the `30-second` grace window.
- [ ] **Eliminate Cross-Cloud Model-Weight Egress Fees via Regional Zero-Egress Storage (`Cloudflare R2`) + VPC-Local `Dragonfly P2P` Caches**: Never pull 100+ GB model weights over public inter-cloud WAN links during elastic bursts.
