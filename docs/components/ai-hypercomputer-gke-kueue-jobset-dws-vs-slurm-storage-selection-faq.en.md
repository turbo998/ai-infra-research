# AI Hypercomputer Cluster Control Plane (`GKE Kueue TAS + JobSet + DWS + Multislice` vs. `Slurm Cluster Toolkit`), 5 Capacity Procurement Modes & 3-Tier Training Storage (`Hyperdisk ML / GCS FUSE / Lustre + Orbax`) Selection Guide

> **Target Audience**: Cloud & AI Infrastructure Solution Architects, AI Platform Engineering Leaders, and Principal SREs architecting `1,024` to `10,000+` chip TPU (`v5p / v6e / TPU7x Ironwood / 8t & 8i`) and GPU (`H200 / B200 / GB200`) training and inference clusters.
> **Core Question**: When operating a **`1,024`-chip AI Hypercomputer writing `4.2 TB` distributed checkpoints every 30 minutes**, why does a naive "Vanilla Kubernetes Default Scheduler on Spot + Uncached Synchronous `GCS FUSE` Checkpointing (Tier 1)" setup suffer **`28.0%` partial-gang allocation deadlocks (`220/256` pods sitting idle), `185-second` cluster-wide checkpoint freezes, and `54.5%` end-to-end cluster Goodput (`$2.48M/month` TCO)**? How should architects choose between **`GKE (Kueue 4-Object Model + Topology-Aware Scheduling [TAS] + JobSet + XPK)`** and **`Slurm (Cluster Toolkit)`**, and how does pairing **`Specific Reservation + COMPACT` Placement + `DWS Flex-Start` (`10-minute` consumption window & `7-day` rotation) + `Hyperdisk ML` (`1.2 TB/s` weight broadcast) + Local-NVMe-Cached `GCS FUSE` + `Orbax` Two-Phase Async & Emergency Checkpointing (Tier 4)** cut checkpoint stalls to **`2.4 seconds`**, eliminate gang deadlocks (**`0.0%`**), boost cluster Goodput to **`96.8%`**, and reduce monthly TCO by **`60.5%` (`$2.48M -> $980K/month`)**?

![AI Hypercomputer Control Plane: GKE (Kueue TAS + JobSet + DWS) vs. Slurm & 3-Tier Storage Summary](../../assets/svg/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

At thousand-chip scale, **your control plane and storage stack dictate whether your accelerators spend 97% of their wall-clock computing gradients or lose 45% of their life waiting on partial gang allocations, 10-minute image-pull timeouts, cross-subblock network hops, and synchronous object-storage checkpoint writes**:

| Control Plane & Storage Tier | Best-Fit Scenario | End-to-End Cluster `Goodput` | `4.2 TB` Checkpoint Stall | Gang Deadlock / Timeout Waste | Monthly `1,024`-Chip Compute & Storage TCO | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Vanilla K8s Default Scheduler (`Spot`) + Uncached Sync `GCS FUSE` (Anti-Pattern)** | ❌ **Never use for multi-host or Multislice training** | **`54.5%` (45.5% wasted!)** | **`185.0 sec`** | **`28.0%` (partial-gang deadlock)** | `$2,480,000 / mo` | ❌ **Reject**: Pod-by-pod scheduling deadlocks; sync GCS FUSE stalls 1,024 chips |
| **Tier 2: Standalone `Slurm (Cluster Toolkit)` + 24x7 Always-On `Managed Lustre`** | Purely batch HPC research labs with legacy `#SBATCH` scripts & zero serving | `86.4%` | `28.0 sec` | `4.5%` | `$1,790,000 / mo` | ✅ **HPC Niche**: Clean `#SBATCH` UX, but siloed from K8s inference pools |
| **Tier 3: `GKE JobSet + DWS Flex-Start` (No `Kueue TAS` Subblock Pinning or Emergency Ckpt)** | Small single-slice fine-tuning jobs | `81.2%` | `42.0 sec` | `6.2%` (misses 10m DWS window!) | `$1,460,000 / mo` | ⚠️ **Fragile**: Cross-subblock placement adds `+22%` collective lag; cold pulls time out |
| **Tier 4: `GKE Kueue TAS + JobSet + DWS/Reservation` + `Hyperdisk ML` + `Orbax` Async/Emergency Ckpt (Recommended)** | **Enterprise Shared Training/RL/Inference & Multislice Production Standard** | **`96.8%`** | **`2.4 sec` (`77x` faster)** | **`0.0%` (all-or-nothing gang)** | **`$980,000 / mo` (`-60.5%`)** | 🏆 **Production Gold Standard**: Subblock-pinned gang, 3-tier storage, `<5s` preempt save |

### 💡 Three Golden Rules for AI Hypercomputer Architects
1. **Never Confuse a Financial Discount (`CUD`) with Physical Capacity Assurance (`Reservation + COMPACT`), and Respect `DWS Flex-Start`'s `10-Minute` & `7-Day` Rules**:
   * **Committed Use Discounts (`1-Yr / 3-Yr CUDs`)** are a billing construct; without a linked **Specific Reservation** and a same-zone **`COMPACT` placement policy**, your job can still hit `ZONE_RESOURCE_POOL_EXHAUSTED` or get scattered across distant network subblocks.
   * For burst training and fine-tuning, **Dynamic Workload Scheduler (`DWS Flex-Start`)** queues a `ProvisioningRequest` until 100% of the requested TPU slice or GPU gang can be allocated atomically. **Two critical production guardrails apply to `DWS Flex-Start`**:
     1. **`10-Minute Consumption Window`**: Once DWS grants the nodes, your pods **must pull their container images and begin running within `10 minutes`**, or the capacity grant is revoked! Mandate **GKE Image Streaming / artifact pre-caching** so 25 GB container images start in **`<35 seconds`**.
     2. **`7-Day Maximum Runtime (`maxRunDurationSeconds <= 604800`)` & Dual Annotations**: DWS Flex-Start nodes terminate at the 7-day mark; configure automated `Orbax` checkpointing and `JobSet` seamless re-queueing before `Day 6, Hour 23`, and include both `provreq.kueue.x-k8s.io/maxRunDurationSeconds` and `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"`.
2. **Orchestrate Multislice Jobs via the `Kueue 4-Object Model + Topology-Aware Scheduling (TAS) + JobSet` and Lock the `7-Layer Software Stack`**:
   * **`Kueue` 4-Object Hierarchy**: `ResourceFlavor` (`Reservation COMPACT`, `DWS Flex-Start`, `Spot`) $\rightarrow$ `ClusterQueue` (cross-team `Cohort` fair-sharing & tidal preemption) $\rightarrow$ `LocalQueue` (namespace queue) $\rightarrow$ `Workload / ProvisioningRequest`.
   * **`JobSet` + `Kueue TAS`**: `JobSet` manages multi-slice `ReplicatedJobs` with coordinated group restarts and deterministic DNS (`<jobset>-<replicatedjob>-<index>-<pod>.<subdomain>`), while **`Kueue TAS`** packs pods strictly onto `cloud.google.com/gce-topology-block / subblock / host` boundaries.
3. **Match Each I/O Pattern to the Right Storage Tier (`Hyperdisk ML` + Local-NVMe-Cached `GCS FUSE` + `Managed Lustre` + `Orbax` Async & Emergency Checkpointer)**:
   * Broadcast read-only base model weights across hundreds of VMs at **`1.2 TB/s` via `Hyperdisk ML`**, stream sharded training datasets via **`GCS FUSE` backed by node-local `NVMe SSD` read caches**, reserve **`Managed Lustre`** for heavy POSIX random I/O or Slurm home directories, and use **`Orbax` Two-Phase Async Checkpointing (`2.4s` stall) + Local-NVMe `Emergency Checkpointer` (`<5s` on preemption)**.

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: How do the 5 capacity procurement models (`Spot`, `DWS Flex-Start`, `DWS Calendar`, `Reservation + COMPACT`, `CUD`) fit together in an enterprise portfolio?
* **The 5-Mode AI Hypercomputer Capacity Matrix**:

| Procurement Mode | Physical Delivery & Scheduling Semantics | Production Gotchas & Guardrails | Target Workload Tier |
| :--- | :--- | :--- | :--- |
| **`Spot VMs / TPUs`** | Deepest discount (`60%–70%` off); immediate opportunistic allocation; `30s` preemption notice | No gang atomicity; frequent preemptions on large slices | Short exploratory runs, stateless batch inference, fault-tolerant jobs with `<5s` local NVMe emergency checkpoints |
| **`DWS Flex-Start`** | Queues request via `ProvisioningRequest` until **100% of the slice/gang is allocated atomically**; much lower preemption than Spot | (1) **Dual annotations required**; (2) **`10-minute` pod consumption window** (enable Image Streaming!); (3) **`7-day` (`604,800s`) max runtime rotation** | 1-hour to 7-day multi-node fine-tuning, RLHF/GRPO post-training, and benchmark sweeps |
| **`DWS Calendar Mode`** | Reserves guaranteed future capacity blocks (`1–90 days`) for a scheduled start date | Requires advance calendar booking | Scheduled release sprints, quarterly model refreshes, customer PoC windows |
| **`Specific Reservation + COMPACT`** | **Guaranteed physical capacity** locked in a single zone with `COMPACT` subblock/cube placement | Billed whether idle or busy (backfill idle hours via `Kueue Cohort` preemption!) | Multi-week frontier pre-training & latency-critical online serving base load |
| **`1-Yr / 3-Yr CUD`** | **Billing discount (`37%–55%+`)**; must be paired with a **Specific Reservation** to guarantee physical availability | Buying a CUD *without* a linked Reservation does NOT prevent `STOCKOUT`! | Steady-state 24x7 baseline fleet |

### Q2: When should an enterprise choose `GKE (Kueue TAS + JobSet + XPK)` vs. `Slurm (Cluster Toolkit)`, and what is the 7-Layer Stack Compatibility Rule?
* **`GKE` vs. `Slurm` Decision Boundary**:
  * Choose **`Slurm (Cluster Toolkit)`** when a pure research team relies 100% on legacy `#SBATCH` scripts, shared POSIX `/home` on `Managed Lustre`, and runs zero Kubernetes online inference microservices.
  * Choose **`GKE + Kueue TAS + JobSet`** (optionally driven by Google's **`xpk` [Accelerated Processing Kit]** CLI for researcher UX) when unifying training, RL, and online `vLLM` serving on a shared cluster with `Kueue Cohort` quota borrowing and automated node auto-repair.
* **The 7-Layer Multi-Host Stack Compatibility Matrix**:
  * Across all hosts in a slice, hash-lock: **(1) Hardware/OCS/NIC Firmware** $\leftrightarrow$ **(2) Host OS Kernel & VFIO/gVNIC** $\leftrightarrow$ **(3) `libtpu.so` / NVIDIA Driver** $\leftrightarrow$ **(4) `ICI` / `NCCL + gIB/TCPXO`** $\leftrightarrow$ **(5) `XLA` / `CUDA`** $\leftrightarrow$ **(6) `JAX` / `PyTorch-XLA`** $\leftrightarrow$ **(7) `MaxText / vLLM-TPU / Orbax`**.

---

## 3. 🧭 Visual Diagrams (Mermaid Decision Tree + Reference Architecture + Sequence Diagram)

### 3.1 AI Hypercomputer Control Plane, Procurement & Storage Decision Tree

```mermaid
flowchart TD
    A["1,024+ Chip AI Hypercomputer Architecture Design"] --> B{"Team Workflow & Fleet Sharing Model?"}
    B -- "Pure HPC batch lab / 100% legacy #SBATCH & POSIX /home" --> C["Control Plane: Slurm (Cluster Toolkit) + Managed Lustre + GCS Archive"]
    B -- "Unified Training + RL + Online Serving / Multi-Team Quota Borrowing" --> D["Control Plane: GKE + Kueue (4-Object Model + TAS) + JobSet + XPK (Tier 4)"]

    D --> E{"Workload Duration & SLA Tier?"}
    E -- "Multi-Week Frontier Pre-Training & Production Serving Base" --> E1["Specific Reservation + Same-Zone COMPACT Policy + 1/3-Yr CUD"]
    E -- "1-Hour to 7-Day Fine-Tuning / RLHF / Evaluation Sprints" --> E2["DWS Flex-Start (Image Streaming <35s for 10m Window + Day-7 Auto-Rotate)"]
    E -- "Preemptible Backfill & Batch Inference" --> E3["Spot Node Pools + Kueue Cohort Idle-Quota Borrowing"]

    E1 & E2 & E3 --> F["3-Tier Storage & Checkpointing:\n(1) Hyperdisk ML (1.2 TB/s Base Weight Broadcast)\n(2) GCS FUSE + Local NVMe SSD Cache (Sharded Corpora)\n(3) Orbax 2-Phase Async (2.4s) + Local NVMe Emergency Checkpointer (<5s)"]
```

### 3.2 Tier 4 `GKE Kueue TAS + JobSet + DWS` & 3-Tier Storage Reference Architecture

```mermaid
flowchart LR
    subgraph Kueue["🎛️ 1. GKE + Kueue TAS + JobSet Control Plane"]
        LQ["LocalQueue (Team Namespace)\nSubmits Multislice JobSet"] --> CQ["ClusterQueue (Cohort Borrowing)\n+ Kueue TAS Topology Scheduler"]
        CQ --> RF1["ResourceFlavor #1: Base Pool\nReservation + COMPACT Subblock"]
        CQ --> RF2["ResourceFlavor #2: Burst Pool\nDWS Flex-Start (10m Window / 7d Max)"]
    end

    subgraph Compute["⚡ 2. Topology-Pinned Compute Slices"]
        RF1 & RF2 ==>|"All-or-Nothing Atomic Grant"| SLICE["JobSet ReplicatedJobs (Slice #1..#4)\n7-Layer Stack Locked + Image Streaming (<35s)"]
    end

    subgraph Storage["💾 3. 3-Tier Storage & Orbax Checkpoint Loop"]
        HD["Hyperdisk ML (Read-Only Multi-Attach)\n1.2 TB/s Base Weight Broadcast"] ==> SLICE
        NVME["Node-Local NVMe SSD\n(1) GCS FUSE Cache (2) <5s Emergency Ckpt"] <==> SLICE
        NVME -.->|"Orbax Background Parallel Composite Upload\n(Only 2.4s Main-Thread Stall)"| GCS["Google Cloud Storage (GCS)\nDurable Multi-Slice Archive"]
    end
```

### 3.3 End-to-End `DWS Flex-Start` 10m Grant, `Orbax` 2.4s Async Save & Preemption Recovery Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Team as ML Team (xpk / kubectl)
    participant Kueue as GKE Kueue TAS + DWS Controller
    participant Nodes as 4x256-Chip Multislice Nodes (Local NVMe)
    participant Storage as Hyperdisk ML + GCS

    Team->>Kueue: Submits 1,024-chip `JobSet` (`maxRunDurationSeconds: "604800"`, `exclusive-topology: subblock`)
    Kueue->>Kueue: Issues `ProvisioningRequest` to `DWS Flex-Start` for 4 contiguous subblock slices
    Kueue->>Nodes: `DWS` atomically provisions all 1,024 chips -> Starts `10-minute` consumption timer
    Storage->>Nodes: `GKE Image Streaming` + `Hyperdisk ML` (`1.2 TB/s`) boots containers & mounts base weights in `32s` (`<10m`!)
    Nodes->>Nodes: Every 30m, `Orbax` copies `4.2 TB` state to Host Pinned RAM / Local NVMe in `2.4s`; training resumes immediately
    Nodes--)Storage: Background threads flush `4.2 TB` shards to `GCS` via parallel composite uploads
    Note over Kueue,Nodes: On host maintenance notice or approaching Day-7 (604,800s) DWS rotation limit
    Nodes->>Nodes: `Orbax Emergency Checkpointer` flushes latest step to Local NVMe in `4.1s`
    Kueue->>Nodes: `JobSet` seamlessly restarts slice from latest step (`96.8%` end-to-end Goodput)
```

---

## 4. 📊 4-Tier Architecture Decision Matrix

| Evaluation Dimension | Tier 1: Vanilla K8s (`Spot`) + Sync `GCS FUSE` | Tier 2: Standalone `Slurm` + Always-On `Lustre` | Tier 3: `GKE JobSet + DWS` (No `TAS` / No Emergency Ckpt) | Tier 4: `GKE Kueue TAS + JobSet + DWS/Res` + `Hyperdisk ML` + `Orbax` Async (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **End-to-End Cluster `Goodput`** | **`54.5%` (deadlocks + sync I/O stalls)** | `86.4%` | `81.2%` | **`96.8%` (Best-in-class effective compute yield)** |
| **`4.2 TB` Checkpoint Stall Time** | **`185.0 sec` (blocks all 1,024 chips)** | `28.0 sec` | `42.0 sec` | **`2.4 sec` (`Orbax` 2-phase RAM/NVMe async flush, `77x` faster)** |
| **Gang Deadlock & `10m DWS` Timeout Waste** | **`28.0%` (partial-gang stranding)** | `4.5%` | `6.2%` (cold pulls exceed 10m window) | **`0.0%` (atomic `Kueue` grant + `Image Streaming <35s`)** |
| **Physical Fabric Locality (`Subblock`)** | Random scatter across zone | Static script mapping | Scattered across subblocks (`+22%` lag) | **✅ `Kueue TAS` + `COMPACT` pins slices inside subblocks/cubes** |
| **Training/Serving Pool Sharing** | ❌ No cohort preemption | ❌ Siloed from K8s serving | ⚠️ Basic queueing | **✅ `Kueue Cohort` borrows idle inference capacity overnight** |
| **Monthly `1,024`-Chip Compute & Storage TCO** | `$2,480,000 / mo` | `$1,790,000 / mo` | `$1,460,000 / mo` | **`$980,000 / mo` (`-60.5%` savings)** |

---

## 5. 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Report

### 5.1 Baseline Workload (`1,024`-Chip AI Hypercomputer / `4.2 TB` Checkpoint Every 30 Minutes)
* **Workload Profile**: Enterprise operating a `1,024`-chip training and post-training cluster (`4` x 256-chip slices) saving `4.2 TB` distributed checkpoints every `30 minutes`.
* **Reproducible Model & Dataset**: See [`experiments/ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection/run.py`](../../experiments/ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection/run.py) and [`results.json`](../../experiments/ai_hypercomputer_gke_kueue_jobset_dws_vs_slurm_storage_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive Sizer**: 📊 [Open AI Hypercomputer Control Plane & 3-Tier Storage Sizer (`ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-sizer.html`)](../reports/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-cards.html`)](../reports/ai-hypercomputer-gke-kueue-jobset-dws-vs-slurm-storage-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Vanilla K8s Spot + Sync GCS", "goodput_pct": 54.5, "ckpt_stall_sec": 185.0, "deadlock_waste_pct": 28.0, "monthly_tco_m": 2.48 },
      { "tier": "Tier 2: Standalone Slurm + Lustre", "goodput_pct": 86.4, "ckpt_stall_sec": 28.0, "deadlock_waste_pct": 4.5, "monthly_tco_m": 1.79 },
      { "tier": "Tier 3: GKE JobSet + DWS (No TAS)", "goodput_pct": 81.2, "ckpt_stall_sec": 42.0, "deadlock_waste_pct": 6.2, "monthly_tco_m": 1.46 },
      { "tier": "Tier 4: Kueue TAS + DWS + Orbax Async", "goodput_pct": 96.8, "ckpt_stall_sec": 2.4, "deadlock_waste_pct": 0.0, "monthly_tco_m": 0.98 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "goodput_pct": "Percentage",
    "ckpt_stall_sec": "Quantity",
    "deadlock_waste_pct": "Percentage",
    "monthly_tco_m": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 96.8% Cluster Goodput, 2.4s Checkpoint Stalls, and Cuts Monthly TCO by 60.5%",
    "subtitle": "Monthly 1,024-chip effective compute and storage TCO ($ USD Millions)",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_m",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ Architecture Review & Production Readiness Checklist

- [ ] **Paired `Specific Reservation + COMPACT Placement` with `1-Yr / 3-Yr CUDs`**: Are baseline `CUD` commitments bound to a same-zone `Specific Reservation` with a `COMPACT` placement policy so capacity is both physically guaranteed and topology-contiguous?
- [ ] **`DWS Flex-Start` Dual Annotations, `<35s` Image Streaming & `7-Day` Auto-Rotation**: Do all `DWS Flex-Start` workloads include `provreq.kueue.x-k8s.io/maxRunDurationSeconds` + `safe-to-evict: "false"`, use `GKE Image Streaming` to start well inside the `10-minute` consumption window, and checkpoint/re-queue cleanly before the `7-day` limit?
- [ ] **`Kueue TAS` Subblock Pinning & `JobSet` Coordinated Recovery**: Does `Kueue TAS` enforce `cloud.google.com/gce-topology-subblock` (or full TPU Cube) affinity across `JobSet` ReplicatedJobs?
- [ ] **Immutable 7-Layer Software Stack Digest**: Are `libtpu / NVIDIA Driver`, `ICI / NCCL / gIB`, `XLA`, `JAX / PyTorch`, and `Orbax` versions pinned to an identical container image digest across all multi-host workers?
- [ ] **3-Tier Storage (`Hyperdisk ML + Local-NVMe-Cached GCS FUSE + Orbax Async/Emergency`)**: Are read-only base weights served from `Hyperdisk ML`, training shards cached on node-local NVMe SSDs via `GCS FUSE`, and `4.2 TB` checkpoints written asynchronously via `Orbax` (`<3s` stall)?
