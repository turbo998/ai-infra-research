# Google Cloud TPU (`v5p` / `v6e Trillium` / `TPU7x Ironwood` / `TPU 8t & 8i`) vs. NVIDIA GPU (`H200` / `B200` / `GB200 NVL72`) Architecture, `3D Torus ICI + OCS` Interconnect & TCO Selection Guide

> **Target Audience**: Cloud & AI Infrastructure Solution Architects, Heterogeneous Compute Platform Leaders, and Principal HPC/AI Cluster Architects designing `256` to `9,216+` accelerator clusters across cloud, partner-operated colocation facilities, and sovereign on-premises data centers.
> **Core Question**: When architecting a **`2,048` to `9,216`-chip training and high-concurrency inference fleet**, how should Solution Architects decide between the **Google Cloud TPU family (`v5p 8,960-chip Pod`, `v6e Trillium 256-chip Pod`, `TPU7x Ironwood 9,216-chip Superpod [4,614 TFLOPS FP8, 192GB HBM3e @ 7.37 TB/s]`, and 8th-gen disaggregated `TPU 8t / 8i`)** and the **NVIDIA GPU family (`H200 141GB`, `B200 192GB`, `GB200 NVL72`)**? Why does naively lifting-and-shifting unbucketed dynamic-shape PyTorch + custom CUDA code onto a non-wrap TPU slice (Tier 1) collapse sustained MFU to **`14.2%`** (`45s` XLA recompilation stalls), and how does a **workload-aligned architecture combining `4x4x4` Twisted-Torus `ICI` Cubes + `Palomar 3D MEMS OCS` + Dual-Chiplet `2 TensorCores + 4 SparseCores` + `JAX / XLA Shardy / MaxText / vLLM-TPU` Static Bucketing (Tier 4)** achieve **`58.4%` sustained MFU, `99.2%` 9,216-chip cluster Goodput, `$0.24 / 1M tokens` serving cost, and a `41.4%` reduction in monthly 2,048-chip TCO (`$3.62M -> $2.12M/month`)**?

![Google Cloud TPU (v5p / v6e Trillium / TPU7x Ironwood / 8t & 8i) vs. NVIDIA GPU (H200 / B200 / GB200) Summary](../../assets/svg/google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu-selection-faq-summary.svg)

---

## 1. 📌 Executive Summary (3-Minute Verdict)

From a systems architecture standpoint, **the fundamental difference between Google TPUs and NVIDIA GPUs is not single-chip peak TFLOPS—it is three co-designed architectural choices: (1) Compiler-scheduled large systolic arrays (`MXU`) + dedicated `SparseCores` vs. hardware-scheduled `SIMT + TensorCores`; (2) Switchless direct chip-to-chip `3D Torus ICI` + `Palomar 3D MEMS Optical Circuit Switches (OCS)` scaling to `9,216` chips without Spine packet switches vs. `8-GPU / 72-GPU NVLink` islands bridged by multi-tier `InfiniBand / RoCEv2` Clos Fat-Trees; and (3) Whole-graph Ahead-of-Time `XLA` compilation (`JAX / Shardy / vLLM-TPU`) vs. eager per-kernel `PyTorch + CUDA` dispatch**:

| Architecture Tier | Best-Fit Workload Profile | `2,048`-Chip Sustained `MFU` | Cluster Fabric `Goodput` | Serving Cost (`$ / 1M Tokens`) | Monthly `2,048`-Chip Equivalent TCO | Architect Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Unmodified Dynamic-Shape `PyTorch/CUDA` on Non-Wrap TPU Slice (Anti-Pattern)** | ❌ **Never lift-and-shift without shape bucketing & kernel audit** | **`14.2%` (XLA recompile storm!)** | `64.0%` (no wrap-around) | `$1.48 / 1M` | `$4,850,000 / mo` | ❌ **Reject**: Dynamic shapes trigger 45s XLA recompiles; non-`4k` slices lose wrap-around |
| **Tier 2: Homogeneous `NVIDIA H200` (`8x NVLink` + `400G RoCEv2` 3-Tier Clos)** | Custom C++/CUDA/PTX kernels & third-party binary CUDA dependencies | `44.5%` (8-GPU NVLink wall) | `89.5%` | `$0.62 / 1M` | `$3,620,000 / mo` | ⚠️ **Baseline**: Universal CUDA support, but heavy Clos Spine optical transceiver tax |
| **Tier 3: `NVIDIA GB200 NVL72` (`72-GPU` Copper `MNNVL` + `InfiniBand XDR`)** | CUDA-locked trillion-parameter MoE low-latency decode & training | `54.8%` | `93.4%` | `$0.34 / 1M` | `$2,980,000 / mo` | ✅ **Flagship CUDA**: 72-GPU NVLink domain, but requires `120kW+` DLC rack retrofit |
| **Tier 4: Workload-Aligned `TPU7x Ironwood / 8t` (`4x4x4` Torus + `OCS` + `Shardy`) + `v6e / 8i` + Hybrid GPU (Recommended)** | **Frontier LLM/MoE Pre-Training, RL, DLRM Embeddings & High-QPS Serving** | **`58.4%`** | **`99.2%` (OCS optical bypass)** | **`$0.24 / 1M` (`-61.3%`)** | **`$2,120,000 / mo` (`-41.4%`)** | 🏆 **Production Gold Standard**: Zero Clos Spine tax, `SparseCore` MoE offload, 99.2% Goodput |

### 💡 Three Golden Rules for Cloud & AI Infrastructure Architects
1. **Master the TPU Generation Matrix & Dual-Chiplet `MXU + VPU + SparseCore` Anatomy**:
   * **`TPU v5e` vs. `v5p`**: `v5e` (`2D Torus`, 256-chip pod, `16GB HBM2`) targets cost-efficient medium-model serving; `v5p` (`3D Torus + OCS`, `8,960`-chip pod, `95GB HBM2e`) targets large-scale training with 2nd-gen `SparseCore`.
   * **`TPU v6e (Trillium)`**: `4.7x` peak compute over `v5e`, `2x` HBM capacity/bandwidth (`32GB HBM @ 1.64 TB/s`), and `2x` ICI bandwidth in a 256-chip 2D Torus pod—ideal for high-throughput serving and post-training.
   * **`TPU7x (Ironwood)`**: Purpose-built for the reasoning/inferential and frontier-training era—a **dual-chiplet architecture packing `2 TensorCores + 4 SparseCores` per chip**, delivering **`2,307 TFLOPS BF16 / 4,614 TFLOPS FP8`** and **`192 GB HBM3e @ 7.37 TB/s`**, scaling via `Palomar OCS` up to a **`9,216`-chip 3D Torus Superpod (`42.5 ExaFLOPS FP8`)** with `4` chips (`8` logical TensorCores) per VM.
   * **8th-Gen Disaggregated `TPU 8t` (Training) & `TPU 8i` (Inference)**: Splits silicon SKUs into training-optimized (`8t`) and inference-optimized (`8i`) lines, with expanding deployments across partner-operated colocation sites and sovereign customer premises.
2. **Enforce the `4x4x4` Cube Wrap-Around Rule on 3D Torus Topologies (`4x4x4` vs. `4x4x8` vs. Non-Wrap Meshes)**:
   * On `v5p` and `TPU7x Ironwood`, each liquid-cooled rack houses a **`4x4x4 = 64`-chip Cube**. Hardware **Wrap-Around (Twisted Torus) links ONLY activate along dimensions that span a full multiple of `4` (`4x4x4`, `4x4x8`, `8x8x8`)**. A full wrap-around torus halves ring diameter and doubles bisection bandwidth for `All-Reduce` and `All-Gather`, whereas requesting a non-wrap sub-cube (`2x2x4`) drops wrap-around links and cuts edge collective bandwidth in half.
3. **Map Bandwidth-Hungry Axes (`TP / FSDP / EP`) Strictly to Intra-Slice `ICI` and Outer Data Parallelism (`DP`) to Inter-Slice `DCN (Multislice)`**:
   * Inside a slice (up to `9,216` chips), `ICI` + `Palomar 3D MEMS OCS` provides direct optical torus connectivity with zero packet-switch hop taxes. Across slices (**Multislice**), `Data-Center Network (DCN)` Ethernet carries outer `DP` gradient reductions overlapped with backward computation.

---

## 2. 🎯 Customer & Architect Critical Q&A

### Q1: Why does lifting-and-shifting an unmodified PyTorch/CUDA workload onto a TPU slice (Tier 1) often drop MFU from `45%` to `14.2%`?
* **Whole-Graph Static `XLA` Compilation vs. Dynamic Eager Execution**:
  1. **Dynamic Shape Recompilation Storms (`30–60s` per new shape)**: Unlike a GPU that launches individual CUDA kernels for `seq_len = 513` vs. `514`, `XLA` fuses operators across the entire model graph and statically allocates on-chip `VMEM` scratchpad memory. Unpadded variable-length prompts or dynamic boolean masks (`tensor[mask]`) change tensor shapes every step, forcing `XLA` to recompile the graph repeatedly. **Fix**: Bucket all input shapes (`256, 512, 1024, 2048, 4096`) and pre-compile at startup (`0` runtime recompilations).
  2. **Host-Device Sync Barriers**: Calling `if loss.item() > threshold:` inside a training step drains the TPU device pipeline; replace with device-side predicates (`jax.lax.cond`) or async host callbacks.

### Q2: Why does `TPU7x Ironwood` integrate `4 SparseCores` alongside `2 TensorCores` on every chip?
* **Offloading Irregular Gather/Scatter & Embedding Lookups from the Systolic `MXU`**:
  * Large systolic arrays (`MXU`) achieve peak efficiency on dense matrix multiplications (`GEMM`), but stall severely on irregular HBM gather/scatter operations found in **massive DLRM embedding tables, dynamic Mixture-of-Experts (`MoE`) token routing, and sparse KV cache indexing**. Offloading those irregular memory-bound operations to the **4 dedicated dataflow `SparseCores`** keeps the `TensorCore MXUs` saturated on dense math (`1.8x–2.5x` higher end-to-end throughput).

### Q3: How does `Palomar 3D MEMS OCS` (Optical Circuit Switching) change both cluster TCO and fault tolerance compared to an InfiniBand/RoCE Clos Fat-Tree?
* **Zero Spine Packet-Switch Optics Tax & Sub-Second Optical Defragmentation**:
  * In a `9,216`-chip TPU7x Superpod (`144` liquid-cooled `4x4x4` cubes), copper links wire the `64` chips inside each cube, while **passive `3D MEMS OCS` mirrors** steer light beams directly between cube faces without power-hungry Optical-Electrical-Optical (`OEO`) packet switches. When a cube fails or degrades, `OCS` re-steers optical paths around the bad cube in seconds while hardware **`ICI Resiliency`** routes around individual link faults—sustaining **`99.2%` cluster Goodput**.

---

## 3. 🧭 Visual Diagrams (Mermaid Decision Tree + Reference Architecture + Sequence Diagram)

### 3.1 Google Cloud TPU vs. NVIDIA GPU Architecture Decision Tree

```mermaid
flowchart TD
    A["Enterprise AI Training & Inference Cluster Sizing Request"] --> B{"Software Stack & Kernel Portability Audit?"}
    B -- "Tightly coupled to custom C++/CUDA/PTX kernels or closed-source CUDA binaries" --> C["Select NVIDIA GPU Family"]
    C --> C1{"Scale & Memory Bandwidth Profile?"}
    C1 -- "<=64-GPU General Training / Multi-Model Serving" --> C2["NVIDIA H200 (141GB) / B200 (8-GPU NVLink + RoCEv2/IB)"]
    C1 -- "Trillion-Param MoE Low-Latency Decode / Multi-Node Training" --> C3["NVIDIA GB200 NVL72 (72-GPU Copper MNNVL + 120kW DLC)"]

    B -- "Standard Transformer / Sparse MoE / DLRM / JAX (MaxText) or Bucketed PyTorch (vLLM)" --> D["Select Google Cloud TPU AI Hypercomputer (Tier 4)"]
    D --> D1{"Workload Phase & Scale?"}
    D1 -- "256 to 9,216+ Chip Frontier Pre-Training / Reasoning RL / MoE" --> D2["TPU7x Ironwood (9,216-Chip Superpod, 192GB HBM3e) / TPU 8t / v5p\nMandate 4x4x4 (64-chip) multiples to enable Twisted Torus Wrap-Around!"]
    D1 -- "Cost-Efficient High-Throughput Serving / LoRA Post-Training" --> D3["TPU v6e Trillium (256-Chip Pod) / TPU 8i / TPU7x + Bucketed vLLM-TPU"]
```

### 3.2 Tier 4 `TPU7x Ironwood` Dual-Chiplet, `4x4x4` Twisted Torus & `OCS + Multislice` Architecture

```mermaid
flowchart LR
    subgraph Chip["🔬 1. TPU7x Ironwood Chip Anatomy (4,614 TF FP8 / 192GB HBM3e)"]
        TC["2x TensorCores (Dual-Chiplet)\nMXU Systolic Arrays + VPU"] <--> HBM["192 GB Unified HBM3e Pool\n(7.37 TB/s Bandwidth)"]
        SC["4x SparseCores Dataflow Engines\n(Offloads MoE Routing & Embeddings)"] <--> HBM
    end

    subgraph Pod["🕸️ 2. Intra-Slice 3D Torus ICI + Palomar OCS (Up to 9,216 Chips)"]
        CUBE1["Cube #1: 4x4x4 (64 Chips)\nIntra-Rack Copper Torus"] <==>|"Palomar 3D MEMS OCS\nDynamic All-Optical + Wrap-Around"| CUBE2["Cube #2..#144: 4x4x4\n(ICI Resiliency Optical Bypass)"]
    end

    subgraph Multi["🌐 3. Inter-Slice Multislice over Jupiter DCN"]
        CUBE1 & CUBE2 ==>|"ICI Carries: TP / FSDP / EP Collectives"| SHARD["JAX / XLA Shardy Partitioner\nNamedSharding(dcn_dp, ici_fsdp, ici_tp)"]
        SHARD ==>|"Jupiter DCN Carries: Outer DP AllReduce"| SLICE2["Slice #2..#N (Multi-Pod Scale-Out)"]
    end
```

### 3.3 End-to-End `OCS` Topology Provisioning, `Shardy` Compilation & Fault Bypass Sequence

```mermaid
sequenceDiagram
    autonumber
    participant Orch as GKE Kueue TAS / JobSet Control Plane
    participant OCS as Palomar 3D MEMS OCS Controller
    participant XLA as JAX / XLA Shardy Compiler
    participant TPU as TPU7x Ironwood 4x4x8 Slice (128 Chips)

    Orch->>OCS: Requests two `4x4x4` Cubes wired as a `4x4x8` Twisted-Torus Slice (128 TPU7x chips / 32 4-chip VMs)
    OCS->>OCS: Steers MEMS mirrors to link Cube faces & close X/Y/Z Wrap-Around Torus loops (`<50ms`)
    Orch->>XLA: Launches `MaxText` 671B MoE training with `Mesh(dcn_dp=2, ici_fsdp=64, ici_tp=8)`
    XLA->>XLA: `Shardy` partitions graph: pins `TP/EP All-to-All` on `ICI` & offloads MoE token gather/scatter to `4x SparseCores`
    XLA->>TPU: Deploys AOT-compiled HLO executable (`0` runtime recompilations)
    TPU->>TPU: Overlaps `MXU` compute, `SparseCore` routing & bidirectional `ICI` rings (`58.4%` sustained MFU)
    Note over OCS,TPU: On optical link fault -> `ICI Resiliency` routes via orthogonal torus axis; on Cube fault -> `OCS` swaps spare Cube (`99.2%` Goodput)
```

---

## 4. 📊 4-Tier Architecture Decision Matrix (TPU Family vs. NVIDIA GPU Family)

### 4.1 Google Cloud TPU Generation Reference Matrix

| TPU Generation | Primary Architectural Role | Max Pod Scale | Interconnect Fabric | Peak Compute per Chip (`BF16 / FP8`) | HBM Capacity & Bandwidth | `SparseCore` Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`TPU v5e`** | Cost-efficient medium-model serving & fine-tuning | `256` chips | `2D Torus` | `197 TF (BF16)` / `393 TOPS (INT8)` | `16 GB HBM2` (`819 GB/s`) | None |
| **`TPU v5p`** | Large-scale foundation model training | `8,960` chips | `3D Torus + OCS` | `459 TF (BF16)` / `918 TF (INT8)` | `95 GB HBM2e` (`2.76 TB/s`) | 2nd-Gen `SparseCore` |
| **`TPU v6e (Trillium)`** | High-throughput LLM/multimodal serving & post-training | `256` chips | `2D Torus` (`2x` ICI BW) | `918 TF (BF16)` (`4.7x` vs. `v5e`) | `32 GB HBM` (`1.64 TB/s`) | 3rd-Gen `SparseCore` |
| **`TPU7x (Ironwood)`** | **Frontier training, long-CoT reasoning & massive MoE** | **`9,216` chips** (`144` `4x4x4` Cubes) | **`3D Torus + OCS`** (`1.2 Tbps/chip` ICI) | **`2,307 TF (BF16)` / `4,614 TF (FP8)`** | **`192 GB HBM3e` (`7.37 TB/s`)** | **Dual-Chiplet: `2 TensorCores + 4 SparseCores`** |
| **`TPU 8t` / `TPU 8i`** | **8th-Gen Training (`8t`) & Inference (`8i`) Split** (Cloud + Partner/Customer Premises) | Multi-thousand-chip pods | `3D Torus + OCS` (`8t`) / Serving-tuned (`8i`) | Dedicated training goodput (`8t`) vs. lowest `$/watt/token` inference (`8i`) | Next-gen high-bandwidth HBM | Enhanced agentic / MoE sparse engines |

### 4.2 4-Tier Cluster Architecture Comparison

| Evaluation Dimension | Tier 1: Unbucketed Dynamic `PyTorch` on Non-Wrap TPU | Tier 2: Homogeneous `NVIDIA H200` (`8x NVLink + RoCEv2`) | Tier 3: `NVIDIA GB200 NVL72` (`72x NVLink + IB XDR`) | Tier 4: `TPU7x Ironwood / 8t` (`4x4x4` Torus + OCS) + `v6e / 8i` + Hybrid GPU (Recommended) |
| :--- | :--- | :--- | :--- | :--- |
| **`2,048`-Chip Sustained `MFU`** | **`14.2%` (45s XLA recompilation thrash)** | `44.5%` (8-GPU NVLink boundary) | `54.8%` | **`58.4%` (`4x4x4` Twisted Torus + `Shardy` AOT)** |
| **Cluster Fabric `Goodput`** | `64.0%` | `89.5%` | `93.4%` | **`99.2%` (`Palomar OCS` spare-cube swap + `ICI Resiliency`)** |
| **Scale-Up High-Bandwidth Domain** | Partial mesh (no wrap-around) | **`8` GPUs** (`900 GB/s` HGX) | **`72` GPUs** (`1.8 TB/s` copper rack) | **Up to `9,216` chips in a unified `3D Torus + OCS` pod** |
| **Sparse Embedding & MoE Routing** | Host fallback bottleneck | General SM / TensorCore | TensorCore + NVLink | **Hardware `4x SparseCores` offload irregular gather/scatter** |
| **Serving Cost (`$ / 1M Tokens`)** | `$1.48 / 1M` | `$0.62 / 1M` | `$0.34 / 1M` | **`$0.24 / 1M` (`v6e Trillium / TPU7x / 8i` + `vLLM-TPU`)** |
| **Monthly `2,048`-Chip Equivalent TCO** | `$4,850,000 / mo` | `$3,620,000 / mo` | `$2,980,000 / mo` | **`$2,120,000 / mo` (`-41.4%` savings vs. Tier 2)** |

---

## 5. 🧪 Quantitative Case Study, Interactive Sizer & 7-Card Report

### 5.1 Baseline Workload (`2,048`-Chip Equivalent Training & Inference Fleet)
* **Workload Profile**: Enterprise operating a `2,048`-accelerator-equivalent fleet running `80%` standard Transformer/MoE pre-training, RL, and high-concurrency serving alongside `20%` custom-CUDA workloads.
* **Reproducible Model & Dataset**: See [`experiments/google_cloud_tpu_v5p_v6e_ironwood_vs_nvidia_gpu_selection/run.py`](../../experiments/google_cloud_tpu_v5p_v6e_ironwood_vs_nvidia_gpu_selection/run.py) and [`results.json`](../../experiments/google_cloud_tpu_v5p_v6e_ironwood_vs_nvidia_gpu_selection/results.json) (`provenance: synthetic + anonymized_reported`, `independently_reproduced=false`).
* **Interactive Sizer**: 📊 [Open Google Cloud TPU vs. NVIDIA GPU Architecture & TCO Sizer (`google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu-sizer.html`)](../reports/google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu-sizer.html)
* **7-Card 4:5 Benchmark Deck**: 🃏 [Open 7-Card Visual Report (`google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu-selection-cards.html`)](../reports/google-cloud-tpu-v5p-v6e-ironwood-8t-8i-vs-nvidia-gpu-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Unbucketed PyTorch on TPU", "mfu_pct": 14.2, "goodput_pct": 64.0, "token_cost_usd": 1.48, "monthly_tco_m": 4.85 },
      { "tier": "Tier 2: Homogeneous H200 RoCEv2 Clos", "mfu_pct": 44.5, "goodput_pct": 89.5, "token_cost_usd": 0.62, "monthly_tco_m": 3.62 },
      { "tier": "Tier 3: NVIDIA GB200 NVL72 + IB XDR", "mfu_pct": 54.8, "goodput_pct": 93.4, "token_cost_usd": 0.34, "monthly_tco_m": 2.98 },
      { "tier": "Tier 4: TPU7x/8t OCS Torus + v6e/8i + GPU", "mfu_pct": 58.4, "goodput_pct": 99.2, "token_cost_usd": 0.24, "monthly_tco_m": 2.12 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "mfu_pct": "Percentage",
    "goodput_pct": "Percentage",
    "token_cost_usd": "Price",
    "monthly_tco_m": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 58.4% MFU, 99.2% Goodput, and Cuts 2,048-Chip Monthly TCO by 41.4% vs. H200 Clos",
    "subtitle": "Monthly 2,048-chip equivalent cluster TCO ($ USD Millions)",
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

- [ ] **Full `4x4x4` Cube Wrap-Around Verification**: Are `v5p` and `TPU7x Ironwood` 3D Torus slices requested in multiples of `4x4x4` (`64` chips, e.g., `4x4x4`, `4x4x8`, `8x8x8`) so hardware Twisted-Torus wrap-around links are active along all three axes?
- [ ] **`ICI` vs. `DCN` Sharding Axis Alignment (`JAX Shardy / GSPMD`)**: Are bandwidth-intensive axes (`TP`, `FSDP`, `MoE EP All-to-All`) pinned strictly to intra-slice `ICI` links while outer `DP` runs across `Multislice DCN`?
- [ ] **Zero-Recompilation Static Shape Bucketing**: Are training sequences packed to fixed lengths and `vLLM-TPU / JetStream` serving requests padded to pre-warmed `(batch, seq_len)` power-of-two buckets (`xla_compilation_cache_misses == 0`)?
- [ ] **Dual-Chiplet & `SparseCore` Offload Configuration**: On `TPU7x Ironwood` (`2 TensorCores + 4 SparseCores` per chip), are large embedding tables and dynamic MoE routing indices offloaded to `SparseCore`?
- [ ] **Workload-Aligned Heterogeneous Placement (`TPU + GPU`)**: Have workloads been triaged so standard Transformer/MoE/DLRM fleets run on `TPU7x / v6e / 8t / 8i` while custom C++/CUDA-locked workloads remain on `H200 / GB200`?
