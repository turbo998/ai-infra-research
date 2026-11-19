# Enterprise Model Asset Registry, Supply-Chain Security (`Safetensors / Cosign`) & Multi-Node P2P Weight Distribution (`OCI / Dragonfly`) Selection Guide: Live `HuggingFace` Pulls vs. Monolithic `155GB` Docker Images vs. S3 Star Pulls vs. `OCI Model Spec (ORAS)` + `Cosign` Gate + `Dragonfly / Spegel` P2P

> **Language / 语言**: **English (Current)** | [中文](./enterprise-model-registry-oci-artifact-p2p-distribution-selection-faq.md)  
> **Scope**: End-to-end production reference architecture, AI supply-chain security, and multi-node weight distribution guide for Cloud & AI Solution Architects, Kubernetes Platform Leads, and DevSecOps Engineers.  
> **Provenance**: `synthetic + anonymized_reported` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`). Deterministic evaluation model sealed in [`experiments/enterprise_model_registry_oci_artifact_p2p_distribution_selection/results.json`](../../experiments/enterprise_model_registry_oci_artifact_p2p_distribution_selection/results.json).

![Enterprise Model Asset Registry, Supply-Chain & P2P Distribution Summary](../../assets/svg/enterprise-model-registry-oci-artifact-p2p-distribution-selection-faq-summary.svg)

---

## ① 📌 Executive Summary (3-Minute Architect Verdict)

When an enterprise AI platform manages **95 foundation models and fine-tuned adapters and scales out a `140 GB` model across `32 GPU nodes` simultaneously (`180` rollout and autoscale events/month)**, how model weights are packaged, verified, and distributed determines both **cluster cold-start speed** and **supply-chain security**:
1. **Anti-Pattern #1: Live `huggingface_hub` Pulls on Pod Startup (Tier 1, `1,420s = 23.6 min` 32-Pod Ready Time, `28.4%` Rollout Failure Rate, `100.0%` `Pickle` RCE Risk)**: Having 32 GPU pods download `32 x 140 GB = 4.48 TB` over corporate NAT saturates internet links, triggers `HTTP 429` rate limits, and exposes the cluster to **Arbitrary Remote Code Execution (`RCE`)** if an upstream repo contains a malicious `pytorch_model.bin` (`pickle.loads`) file;
2. **Anti-Pattern #2: Baking `140 GB` Model Weights Inside Monolithic Docker Container Images (Tier 2, `980s` Ready Time, `$124,000/month` TCO)**: Coupling 140 GB of weights with 15 GB of Python/CUDA code inside one `Dockerfile` means a 1-line code fix forces CI/CD to rebuild, push, and extract a **155 GB image**, filling up `/var/lib/containerd` node disks;
3. **Bottleneck #3: Un-Cached S3 / Blob Object Storage Star Pulls (`initContainer` `s3 cp / azcopy`, Tier 3, `295s` Ready Time, `4,480 GB` Cross-AZ Egress)**: Pulling 32 copies of 140 GB in a star topology from one bucket bottlenecks on gateway egress bandwidth while 32 H100 nodes sit idle waiting for network I/O.

Evaluated across a **32-node simultaneous rollout of a `140 GB` model (`180` events/month)**, the four model registry and distribution architectures compare as follows:

| Architectural Tier | Core Packaging & Distribution Stack | 32-Node `140GB` Ready Time | Supply-Chain `Pickle` RCE Risk | Registry / Cross-AZ Egress | Rollout Timeout Failure Rate | Monthly Distribution TCO | Production Verdict |
|---|---|---|---|---|---|---|---|
| **🟥 Tier 1: Live `huggingface_hub` Pull** | Downloads weights from public internet on pod start | `1,420.0s (23.6m)` ✕ | `100.0%` ✕ *(`.bin` pickle RCE)* | `4,480 GB` ✕ | `28.4%` ✕ *(NAT & 429 throttling)* | `$146,000 / mo` ✕ | ⛔ Critical security & availability failure |
| **🟨 Tier 2: Monolithic `155GB` Docker Image** | Couples code + 140GB weights in one `Dockerfile` | `980.0s (16.3m)` ✕ | `35.0%` | `4,960 GB` ✕ *(re-pushes 155GB)* | `16.5%` ✕ *(kubelet disk full)* | `$124,000 / mo` ✕ | ⛔ Breaks CI/CD velocity & node storage |
| **🟦 Tier 3: Private S3 Star Pull (`initContainer`)** | Every pod runs `s3 cp / azcopy` from central bucket | `295.0s (4.9m)` | `12.0%` | `4,480 GB` ✕ *(32x duplicate pull)* | `2.4%` | `$68,000 / mo` | ⚠️ Safe, but star topology bottlenecks at scale |
| **🟩 Tier 4: `OCI Model Spec (ORAS)` + `Safetensors/Cosign` + `Dragonfly / Spegel` P2P** | **Decoupled OCI Model Artifacts (`Harbor/ACR`) + `Safetensors` & `Cosign` Gate + `Dragonfly` P2P Mesh + Read-Only NVMe CSI `mmap`** | **`28.0s` 🏆** *(50x faster)* | **`0.0%` 🏆** *(zero-code tensor + signed)* | **`140 GB` 🏆** *(-96.9% egress)* | **`0.0%` 🏆** | **`$21,000 / mo` 🏆** *(-85.6%)* | 🌟 **Recommended Enterprise Model Registry Standard** |

> 💡 **Golden Rule for Enterprise Model Registry & Distribution Architects**: **"Enforce three decoupled invariants: (1) never bake 100GB+ weights into application Dockerfiles—package weights separately as content-addressable `OCI Model Artifacts (ORAS / CNCF ModelPack)` inside your existing `Harbor / Azure Container Registry`; (2) quarantine all external models through an Airlock that enforces zero-code `safetensors`, blocks `pickle` (`.bin/.pt`), and signs approved digests with `Sigstore Cosign`; and (3) distribute weights across GPU nodes using an in-cluster `Dragonfly / Spegel` P2P mesh so the registry sends `140 GB` only once and all 32 GPU nodes become Ready in `28 seconds`."**

---

## ② 🎯 Customer & Architect Deep-Dive Q&A (3 Model Bloat, P2P Acceleration & Pickle RCE Pain Points)

### Q1: Why should model weights be stored as decoupled `OCI Model Artifacts (ORAS)` in `Harbor / ACR` rather than baked inside Docker images or left as raw S3 folders?
- **Content-Addressable Layer Sharing Without Rebuilding Code Images**:
  - Application code changes daily; 140 GB base weights change monthly. Storing model shards (`model-00001-of-00015.safetensors`) as **OCI Artifacts (`ORAS`)** lets you reuse your existing enterprise container registry (`Harbor`, `ACR`, `ECR`) with RBAC, vulnerability scanning, and immutable `@sha256` digests—while keeping your `vLLM` runtime image at a nimble `12 GB`.

### Q2: How does `Dragonfly / Spegel` P2P distribution cut 32-node rollout time from `295s–1,420s` down to `28.0 seconds` while reducing cross-AZ egress by `96.9%`?
- **Peer-to-Peer 16 MB Chunk Swarming Over Intra-Cluster Networks**:
  - Instead of 32 GPU nodes pulling `32 x 140 GB = 4,480 GB` through the central registry gateway, `Dragonfly` coordinates the 32 nodes to pull distinct `16 MB` chunks from the registry (**only `140 GB` crosses the registry/AZ boundary once**) and immediately exchanges chunks peer-to-peer across the `100G/400G` intra-cluster fabric into a shared read-only NVMe CSI volume (`28.0s` total ready time).

### Q3: How does the Airlock Quarantine Pipeline eliminate `100%` of `Pickle` RCE and weight tampering risks?
- **Zero-Code `Safetensors` + `ModelScan` + `Sigstore Cosign` Admission Control**:
  - Legacy `.bin / .pt` files use Python `pickle`, which allows arbitrary code execution during `torch.load()`.
  - The Airlock converts/verifies all checkpoints as pure **`safetensors`**, scans them with `ModelScan`, signs the OCI digest via **`Sigstore Cosign`**, and enforces signature verification at pod creation via **`Kyverno / OPA Gatekeeper`** with `HF_HUB_OFFLINE=1`.

---

## ③ 🧭 Visual Architecture Diagrams (Decision Tree + OCI/Cosign/Dragonfly Topology + Sequence Flow)

### 3.1 Enterprise Model Asset Registry & Distribution Selection Decision Tree

```mermaid
flowchart TD
    A["Enterprise Model Asset Registry & Distribution Selection<br/>(32 GPU Nodes + 140GB Checkpoint + 180 Rollouts/Month)"] --> B{"Do production pods pull weights directly from<br/>public HuggingFace Hub on startup?"}
    B -- "Yes (Tier 1 Anti-Pattern)" --> C["⛔ 28.4% Rollout Timeouts & 100% Pickle RCE Exposure!<br/>Block public hub egress (HF_HUB_OFFLINE=1) immediately"]
    B -- "No (Internal Storage)" --> D{"How are model weights packaged and distributed to 32 nodes?"}
    D -- "Baked inside a monolithic 155GB Dockerfile (Tier 2)" --> E["⛔ 980s Rollout & Kubelet Disk Exhaustion!<br/>1-line code fix triggers 155GB image rebuild"]
    D -- "32 nodes run s3 cp from central bucket in star topology (Tier 3)" --> F["⚠️ 295s Gateway Bottleneck & 4,480 GB Cross-AZ Egress"]
    D -- "OCI Model Spec (ORAS) + Safetensors/Cosign + Dragonfly P2P (Tier 4)" --> G["🟩 Tier 4: Decoupled OCI Artifact + Cosign Gate + Dragonfly P2P 🏆<br/>(28.0s 32-Node Ready | 140 GB Egress (-96.9%) | 0.0% RCE | -85.6% TCO)"]
    style G fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style C fill:#7f1d1d,stroke:#ef4444,color:#fecaca
    style E fill:#7f1d1d,stroke:#ef4444,color:#fecaca
```

### 3.2 Tier 4 Airlock Quarantine (`Safetensors + Cosign`) + `OCI Model Registry` + `Dragonfly P2P` Architecture

```mermaid
flowchart LR
    subgraph Airlock["1. Airlock Quarantine Pipeline"]
        Ext["External Hub / Internal Training"] --> Scan["🛡️ Format & Security Gate<br/>1. Enforce pure Safetensors (ban pickle .bin)<br/>2. ModelScan + Sigstore Cosign KMS Sign"]
    end

    subgraph Registry["2. Unified OCI Artifact Registry"]
        Scan -->|oras push (@sha256)| Harbor["📦 Harbor / Azure Container Registry<br/>12GB Engine Image decoupled from 140GB Tensor Blobs"]
    end

    subgraph Cluster["3. K8s Admission & Dragonfly P2P Mesh"]
        Harbor -->|Single 140GB Pull Across AZ!| P2P["⚡ Dragonfly / Spegel In-Cluster P2P Mesh<br/>32 GPU nodes exchange 16MB chunks over 100G/400G LAN"]
        P2P --> CSI["💾 Shared Read-Only NVMe CSI Cache (mmap)<br/>Kyverno verifies Cosign signature -> Ready in 28s!"]
    end
```

### 3.3 End-to-End Sequence: Scaling Out a `140 GB` Model Across 32 H100 Nodes in `28 Seconds`

```mermaid
sequenceDiagram
    autonumber
    participant CI as Model Release Pipeline
    participant Reg as Enterprise Harbor / ACR (OCI Registry)
    participant Kyv as K8s Kyverno Admission Webhook
    participant DF as Dragonfly P2P Coordinator
    participant Nodes as 32x H100 GPU Nodes (NVMe CSI Cache)

    CI->>Reg: Push verified `safetensors` OCI Model Artifact & attach `Cosign` signature (`@sha256:8f9c...`)
    Nodes->>Kyv: Request creation of 32 concurrent vLLM pods mounting `@sha256:8f9c...`
    Kyv->>Reg: Verify `Cosign` KMS signature & `safetensors` mediaType -> Approved!
    Nodes->>DF: 32 nodes request 140GB model shards simultaneously
    DF->>Reg: Coordinate striped download so only 1 copy (140GB total) crosses the registry/AZ boundary (-96.9% egress!)
    DF->>Nodes: Peer 16MB tensor chunks across the 32 nodes over 100G/400G LAN into local NVMe CSI cache
    Nodes-->>Nodes: t=28.0s: All 32 pods `mmap` weights directly from NVMe into HBM3 and pass readiness probes!
```

---

## ④ 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | 🟥 Tier 1: Live `HF` Pull | 🟨 Tier 2: Monolithic `155GB` Docker | 🟦 Tier 3: Private `S3` Star Pull | 🟩 Tier 4: `OCI (ORAS)` + `Cosign` + `Dragonfly P2P` |
|---|---|---|---|---|
| **32-Node `140GB` Ready Time** | `1,420.0s` (23.6m) ✕ | `980.0s` (16.3m) ✕ | `295.0s` (4.9m) | 🏆 **`28.0s` (in-cluster P2P mesh + NVMe CSI cache)** |
| **Supply-Chain `Pickle` RCE Risk** | `100.0%` ✕ (`.bin` allowed) | `35.0%` | `12.0%` | 🏆 **`0.0%` (`Safetensors`-only + `Cosign` admission gate)** |
| **Registry / Cross-AZ Egress** | `4,480 GB` ✕ | `4,960 GB` ✕ | `4,480 GB` ✕ | 🏆 **`140 GB` (pulled once per cluster, `-96.9%`)** |
| **Rollout Timeout Failure Rate** | `28.4%` ✕ (429/network) | `16.5%` ✕ (disk full) | `2.4%` | 🏆 **`0.0%` (100% air-gapped internal P2P)** |
| **Monthly Distribution TCO** | `$146,000 / mo` | `$124,000 / mo` | `$68,000 / mo` | **`$21,000 / mo` 🏆 (-85.6%, saves $1.50M/yr)** |

---

## ⑤ 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Visual Report

- **Interactive HTML Sizer**: 👉 [Open Enterprise Model Registry, Supply-Chain & P2P Weight Distribution Sizer](../reports/enterprise-model-registry-oci-artifact-p2p-distribution-sizer.html)
- **7-Card 4:5 Visual Benchmark Report**: 👉 [Open 7-Card Visual Benchmark Deck](../reports/enterprise-model-registry-oci-artifact-p2p-distribution-selection-cards.html)
- **Deterministic Verification**:
  ```bash
  python experiments/enterprise_model_registry_oci_artifact_p2p_distribution_selection/run.py --check
  python -m unittest tests/test_enterprise_model_registry_oci_artifact_p2p_distribution_selection.py
  ```

---

## ⑥ 🛠️ Production Readiness & Architecture Review Checklist

- [ ] **Block Direct Internet Downloads (`HF_HUB_OFFLINE=1`) on All Production GPU Pods**: Serve weights exclusively from an internal registry.
- [ ] **Decouple Engine Container Images (`12 GB`) from Model Weight Artifacts (`140 GB`) Using `ORAS / OCI Model Spec`**: Mount weights via a read-only NVMe CSI volume (`mmap` shared across local processes).
- [ ] **Ban Python `pickle` Checkpoints (`.bin`, `.pt`, `.pkl`) and `trust_remote_code=True` in Production**: Require zero-code `safetensors` / `GGUF` scanned by `ModelScan`.
- [ ] **Sign All Approved Model Artifacts with `Sigstore Cosign` and Enforce Immutable `@sha256` Digest Verification via `Kyverno / Gatekeeper`**.
- [ ] **Deploy `Dragonfly` or `Spegel` P2P Distribution in Multi-Node GPU Clusters**: Eliminate duplicate cross-AZ egress and bring 32-node rollouts under `30 seconds`.
