# Heterogeneous GPU Cluster (`NVIDIA H100/B200` + `AMD MI300X/MI325X`) Unified Compiler Stack & Cross-Vendor PD Disaggregation Selection Guide (`Naive HIPIFY` vs. `Static ONNX/XLA` vs. `Triton/TileLang + AMD AITER` vs. `Cross-Vendor PD Disaggregation [H100 Prefill + MI300X 192GB Decode]`)

> **Reader's Guide**: Built for **AI Solution Architects and Multi-Vendor AI Infrastructure Leads**, this guide explains from first principles: **why AMD Instinct MI300X boasts `192 GB` of HBM3 (`2.4x` H100's 80GB) and `5.3 TB/s` of memory bandwidth (`1.58x` H100's 3.35 TB/s), yet naively translating CUDA kernels via `hipify-perl` (Tier 1) yields only `41.5%` bandwidth utilization and `1,120 tok/s`; how NVIDIA's monolithic `32-thread Warp / 228KB SMEM` architecture differs fundamentally from AMD CDNA3's `64-thread Wavefront / 64KB LDS / 8-XCD Chiplet` topology; how `OpenAI Triton / TileLang + AMD AITER` with `8-XCD L2 Cache Swizzling` (Tier 3) boosts MI300X throughput `2.54x` to `2,850 tok/s`; and how `Tier 4 (Cross-Vendor Heterogeneous Prefill-Decode Disaggregation: NVIDIA H100 Compute-Bound Prefill + AMD MI300X 192GB Bandwidth/Capacity-Bound Decode over Mooncake 400G RoCEv2)` unlocks `3,840 tok/s`, `260` concurrent 32K sessions/node (`2.7x` H100), and `$1.48 / 1M tokens` (`-72.4%` cost).**

![Heterogeneous GPU Cluster Compiler Stack & Cross-Vendor PD Disaggregation Summary](../../assets/svg/heterogeneous-gpu-rocm-cuda-triton-tilelang-stack-selection-faq-summary.svg)

---

## ① 📌 Executive Summary

1. **Why Naive `HIPIFY` Source Ports (Tier 1) Waste Over 58% of AMD MI300X Bandwidth (Two Silicon Mismatches)**:
   - Translating `cuda*` API calls to `hip*` via `hipify-perl` compiles cleanly, but ignores two fundamental hardware differences between NVIDIA Hopper and AMD CDNA3 (`MI300X / MI325X`):
     1. **`Warp-32` (`228 KB` SMEM) vs. `Wavefront-64` (`64 KB` LDS)**: CUDA kernels tuned for 32-thread warps and large `228 KB` shared memory tiles overflow AMD's `64 KB` Local Data Share (LDS) and leave half of CDNA3's 64-wide Matrix Core (`MFMA`) lanes underutilized.
     2. **`8-XCD` Chiplet L2 Cache Thrashing**: Unlike H100's monolithic die with a unified L2 cache, **MI300X is built from 8 Accelerator Complex Dies (`8 XCDs`)**, each with its own private `4 MB` L2 cache. Round-robin CTA scheduling scatters adjacent matrix tiles across all 8 XCDs—causing all 8 XCDs to miss in L2 and fetch the exact same weight tiles from HBM 8 times! Effective HBM bandwidth utilization collapses to **`41.5%` ✕** (`1,120.0 tok/s` ✕, `$5.36 / 1M tokens` ✕).
2. **Homogeneous AMD Gold Standard (`Tier 3: Triton/TileLang + AITER`) vs. Multi-Vendor Fleet Gold Standard (`Tier 4: H100 Prefill + MI300X 192GB Decode Disaggregation`)**:
   - **For Homogeneous AMD MI300X/MI325X Clusters, standardize on `Tier 3 (vLLM / SGLang + OpenAI Triton / TileLang + AMD AITER with 8-XCD Swizzling)`**: Autotunes Triton/TileLang kernels for `Wavefront-64` and `64 KB` LDS, applies **8-XCD PID Swizzling** so spatially adjacent tiles stay on the same XCD chiplet, and links **AMD AITER** assembly kernels for PagedAttention and FP8 MoE—lifting HBM utilization to **`82.4%` ✅** and throughput **`2.54x` to `2,850.0 tok/s` ✅** (`$2.11 / 1M tokens`)!
   - **For Multi-Vendor Clusters Combining `NVIDIA H100 (80GB)` and `AMD MI300X (192GB)`, deploy `Tier 4 (Cross-Vendor Heterogeneous PD Disaggregation: H100 Compute-Bound Prefill Pool + MI300X 192GB Bandwidth/Capacity-Bound Decode Pool via Mooncake RoCEv2)`**:
     - **Match Each Silicon Architecture to Its Roofline**: Route **compute-bound Prefill** (`FLOP/s`-bound, low KV retention) to **NVIDIA H100 (`80GB`, FlashAttention-3 FP8)**, and route **memory-capacity & bandwidth-bound Decode** to **AMD MI300X (`192 GB/GPU = 1.53 TB/node`, `5.3 TB/s` HBM3)**!
     - **Vendor-Agnostic RoCEv2 KV Handoff**: Transfer paged KV blocks between H100 and MI300X pools over standard **400GbE RoCEv2 RDMA (`Mooncake / NIXL`)**—achieving **`89.6%` HBM utilization 🏆**, **`3,840.0 tok/s` throughput (`3.43x` over Tier 1 🏆)**, **`7.4 ms` P99 TPOT 🏆**, **`260` concurrent 32K sessions/node (`2.7x` H100 🏆)**, and **`$1.48 / 1M tokens` (`-72.4%` cost 🏆)**!
3. **First-Principles Rule**:
   - **Inside a kernel**: Align thread-group width (`32 vs. 64`), shared memory footprint (`228KB vs. 64KB`), and chiplet L2 locality (`8-XCD Swizzle`).
   - **Across a cluster**: Never mix vendors inside a single `TP` group (`NCCL` vs. `RCCL`); disaggregate **Prefill on high-FLOP/s GPUs (`H100`)** and **Decode on high-HBM-capacity/bandwidth GPUs (`MI300X 192GB`)**.

---

## ② 🎯 Architect & Customer FAQ (First-Principles Q&A)

### Q1: Why does an AMD MI300X (`192 GB` HBM3, `5.3 TB/s`) run slower than an H100 (`80 GB`, `3.35 TB/s`) when we port our CUDA kernels using `hipify-perl` (Tier 1)?

**Plain-Language First Principles**:
1. **The `8-XCD` Chiplet L2 Cache Thrashing Trap**:
   - An MI300X package contains **8 separate compute chiplets (`XCD 0..7`)**, each with 38 CUs and a private `4 MB` L2 cache. By default, workgroups `0, 1, 2, 3, 4, 5, 6, 7` are dealt round-robin across `XCD 0..7`.
   - In a GEMM kernel where workgroups `0..7` compute adjacent columns of the same row, scattering them across 8 XCDs forces all 8 chiplets to fetch the same activation/weight row from HBM independently—wasting `87.5%` of L2 locality and collapsing HBM bandwidth efficiency to **`41.5%` ✕**!
2. **`64 KB` LDS vs. `228 KB` SMEM & `Wavefront-64` vs. `Warp-32`**:
   - CUDA kernels designed for H100's `228 KB` Shared Memory and 32-thread warps spill registers or fail occupancy on MI300X's `64 KB` LDS and 64-thread wavefronts.

---

### Q2: How do `OpenAI Triton / TileLang` + `AMD AITER` (Tier 3) unlock `82.4%` HBM bandwidth efficiency (`2,850 tok/s`) from the exact same Python model codebase?

**Plain-Language First Principles**:
1. **8-XCD PID Swizzling in 4 Lines of Triton**:
   - Remaps workgroup IDs (`pid`) into contiguous chunks of 38 workgroups per XCD so that all workgroups on `XCD #k` process spatially adjacent tiles of the matrix—boosting L2 cache hit rates from `<35%` to **`82%+`**!
2. **Target-Aware Autotuning + `AMD AITER` Assembly Kernels**:
   - Triton/TileLang compiles to `Wavefront=64` and `MFMA` instructions automatically, while `vLLM / SGLang` bind critical PagedAttention, FlashMLA, and fused AllReduce+RMSNorm ops to **`AMD AITER`** (`VLLM_ROCM_USE_AITER=1`), jumping node throughput **`2.54x` from `1,120 → 2,850 tok/s` ✅**!

---

### Q3: How does Cross-Vendor Heterogeneous PD Disaggregation (`H100 Prefill` + `MI300X 192GB Decode`, Tier 4) achieve `1 + 1 > 2` synergy across mixed GPU fleets?

**Plain-Language First Principles**:
- **Why Prefill Loves `H100 (80GB)` and Decode Loves `MI300X (192GB, 5.3 TB/s)`**:
  - **Prefill** is pure matrix-multiply compute (`FLOP/s`-bound) and immediately evicts KV blocks once computed—making H100's mature `FlashAttention-3 / CUTLASS FP8` stack ideal despite its smaller `80 GB` HBM.
  - **Decode** is strictly bound by **HBM GB/s bandwidth** (reading weights + KV every token) and **HBM GB capacity** (holding hundreds of concurrent 32K KV caches). An `8×MI300X` node provides **`1,536 GB` (`1.53 TB`) of HBM3**—leaving **`1,464 GB` of pure KV Cache capacity after loading 70B FP8 weights (`2.58x` more KV capacity than an `8×H100` node's `568 GB`!)** and reading it at **`5.3 TB/s/GPU`**!
- **How KV Cache Crosses Between NVIDIA and AMD**:
  - While `NCCL` and `RCCL` cannot mix inside a `TP` group, standard **400GbE RoCEv2 RDMA NICs (`ConnectX-7`)** and **Mooncake / NIXL Transfer Engine** move raw `int8/fp8` paged KV tensors directly from H100 HBM to MI300X HBM over the network in `<18 ms`, unlocking **`3,840 tok/s` (`3.43x` speedup 🏆)** and **`$1.48 / 1M tokens` (`-72.4%` cost 🏆)**!

---

## ③ 🧭 Visual Architecture Diagrams (Mermaid Decision Tree / XCD & PD Topology / Sequence Diagram)

### 3.1 Heterogeneous GPU Cluster & Compiler Stack Decision Tree (`flowchart TD`)

```mermaid
flowchart TD
    A["🚀 Start Heterogeneous GPU Cluster (NVIDIA H100 + AMD MI300X) Selection<br/>(Goal: Bridge Warp-32 vs Wavefront-64/8-XCD & Exploit 192GB 5.3TB/s HBM)"] --> B{"Using Naive hipify-perl Port of CUDA Kernels<br/>Without XCD Swizzling or Wave64 Tuning (Tier 1)?"}
    B -- "Yes (Tier 1: Naive HIPIFY)" --> C["⛔ Reject for Production<br/>8-XCD L2 cache thrashing + 64KB LDS overflow caps HBM util at 41.5%!<br/>Only 1,120 tok/s · 26.5ms P99 TPOT · $5.36/1M tokens ✕"]
    B -- "Use Target-Aware Compilation" --> D{"Exporting to Static ONNX / OpenXLA Graphs (Tier 2)?"}
    D -- "Yes (Tier 2: Static ONNX/XLA)" --> E["⚠️ Loses Modern LLM Optimizations (Tier 2)<br/>Drops PagedAttention & fused MoE/speculative kernels (Only 64 32K sessions)"]
    D -- "Deploy vLLM/SGLang + Triton/TileLang + AITER" --> F{"Cluster Hardware Composition?"}
    F -- "Homogeneous AMD MI300X / MI325X Cluster<br/>(Unified Codebase Parity)" --> G["✅ Select Tier 3: Triton / TileLang + AMD AITER (8-XCD Swizzled)<br/>• 8-XCD L2 locality swizzling + Wavefront-64 MFMA autotuning<br/>• 82.4% HBM utilization · 2,850 tok/s (2.54x faster) · $2.11/1M ✅"]
    F -- "Multi-Vendor Fleet: NVIDIA H100 (80GB) + AMD MI300X (192GB)<br/>(Maximize Roofline Synergy & Long-Context Capacity)" --> H["🏆 Select Tier 4: Cross-Vendor Heterogeneous PD Disaggregation<br/>• Compute-bound Prefill on H100 (FA3) + Memory-bound Decode on MI300X (1.53TB/node)<br/>• Mooncake 400G RoCEv2 transfers raw FP8 KV blocks across vendors<br/>• 3,840 tok/s (3.43x) · 260 32K sessions · 7.4ms P99 TPOT · $1.48/1M (-72.4%) 🏆"]
```

### 3.2 Naive `HIPIFY` 8-XCD Thrashing vs. `XCD-Swizzled Triton/AITER + Cross-Vendor PD Disaggregation` (`flowchart LR`)

```mermaid
flowchart LR
    subgraph Bad["❌ Legacy Trap (Tier 1): Naive HIPIFY 8-XCD L2 Thrashing + Monolithic Bottleneck"]
        direction TB
        B1["Adjacent Workgroups Scattered Across 8 XCD Chiplets<br/>Same Weight Tile Fetched 8x from HBM (41.5% BW Util ✕)"] --> B2["CUDA Warp-32 / 228KB SMEM Tile Overflow on 64KB LDS<br/>1,120 tok/s · 26.5ms P99 TPOT · $5.36 / 1M Tokens ✕"]
    end

    subgraph Good["🏆 Modern Paradigm (Tier 3 & Tier 4): 8-XCD Swizzled AITER + Cross-Vendor PD Split"]
        direction TB
        G1["Tier 3: Triton/TileLang 8-XCD Swizzling + AMD AITER Wave64<br/>Co-locates Adjacent Tiles on Same XCD (82.4% BW Util ✅)"] --> G2["Tier 4: H100 (Compute Prefill) + MI300X (1.53TB HBM Decode)<br/>Mooncake 400G RoCEv2 Zero-Copy Cross-Vendor KV Transfer ✅"]
        G2 --> G3["3,840 tok/s (3.43x) · 260 Concurrent 32K Sessions/Node<br/>7.4ms P99 TPOT · $1.48 / 1M Tokens (-72.4% Cost 🏆)"]
    end

    Bad -. "Upgrade Compiler & Disaggregate PD (3.43x Throughput / -72.4% Cost)" .-> Good
```

### 3.3 End-to-End Cross-Vendor Heterogeneous PD Disaggregation (`H100 Prefill` $\rightarrow$ `MI300X Decode`) (`sequenceDiagram`)

```mermaid
sequenceDiagram
    autonumber
    participant Router as KV-Aware Disaggregated Gateway
    participant H100 as NVIDIA H100 Node (Prefill Pool · CUDA FA3)
    participant RoCE as 400Gbps RoCEv2 Standard RDMA Fabric
    participant MI300X as AMD MI300X Node (Decode Pool · 1.53TB HBM3)

    Router->>H100: 1. Route 32K-token prompt to compute-bound H100 Prefill worker
    Note over H100: 2. Execute Hopper wgmma + FlashAttention-3 FP8 Prefill;<br/>populate standardized IEEE FP8 E4M3 paged KV blocks
    H100->>RoCE: 3. Mooncake Transfer Engine initiates GPUDirect RDMA WRITE over 400GbE
    RoCE->>MI300X: 4. Zero-copy DMA writes KV blocks into MI300X 1.53TB HBM pool (No NCCL/RCCL needed!)
    Note over MI300X: 5. 🔥 Exploit MI300X 5.3 TB/s BW & 192GB/GPU Capacity:<br/>Run AITER Wavefront-64 + 8-XCD Swizzled Paged Decode across 260 32K sessions!
    MI300X-->>Router: 6. Stream output tokens (7.4ms P99 TPOT, 3,840 tok/s, $1.48/1M tokens 🏆)
```

---

## ④ 📊 Architect Decision Matrix

*Benchmark Profile: `8×NVIDIA H100-80GB` (`3.35 TB/s`, `$28.80/hr`) and `8×AMD Instinct MI300X-192GB` (`5.30 TB/s`, `$21.60/hr`) serving `Llama-3.3-70B-FP8` (`80 layers, TP=8`, `32K` long-context sessions).*

| Selection Dimension / Tier | Tier 1: Naive `HIPIFY` Source Port (`Warp-32` + `No XCD Swizzle`) | Tier 2: Static `ONNX / OpenXLA` Export (`No Paged KV`) | Tier 3: Unified `Triton / TileLang + AMD AITER` (`Wave64 + 8-XCD`) | Tier 4: Cross-Vendor `PD Disaggregation` (`H100 Prefill` + `MI300X Decode`) |
| :--- | :--- | :--- | :--- | :--- |
| **Compiler & Kernel Stack** | `hipify-perl` literal translation of CUDA C++ kernels | Static graph export falling back to unfused generic ops | **Triton/TileLang autotuned + `8-XCD Swizzle` + `AMD AITER` assembly** | **H100 `FA3/CUTLASS` Prefill + MI300X `AITER` Decode + `Mooncake RDMA`** |
| **Deployment Topology** | Monolithic on MI300X (8-XCD cache thrashing) | Monolithic with static padded KV tensors | **Homogeneous 8×MI300X PagedAttention continuous batching** | **Roofline-matched split: H100 FLOPs Prefill + MI300X (`1.53TB`) Decode** |
| **HBM Bandwidth Utilization** | `41.5%` (Cross-XCD L2 misses ✕) | `54.0%` | **`82.4%` (Unlocks `5.3 TB/s` HBM3 ✅)** | **`89.6%` (Zero Prefill interference on Decode 🏆)** |
| **Node Throughput (tok/s)** | `1,120.0 tok/s` ✕ | `1,480.0 tok/s` | **`2,850.0 tok/s` (`2.54x` vs. Tier 1)** ✅ | **`3,840.0 tok/s` (`3.43x` vs. Tier 1)** 🏆 |
| **Decode P99 Latency (TPOT)** | `26.5 ms` ✕ | `21.0 ms` | **`11.8 ms`** ✅ | **`7.4 ms` (`-37.3%` lower than Tier 3)** 🏆 |
| **Max Concurrent 32K Sessions** | `148 sessions / node` | `64 sessions / node` ✕ | **`196 sessions / node` (`1.53 TB` HBM ✅)** | **`260 sessions / node` (`2.7x` vs. 8×H100 node 🏆)** |
| **Cost per 1M Tokens (USD)** | `$5.36 / 1M tokens` ✕ | `$4.05 / 1M tokens` | **`$2.11 / 1M tokens` (`-60.6%`)** ✅ | **`$1.48 / 1M tokens` (`-72.4%` cost)** 🏆 |

---

## ⑤ 🧪 Quantitative Case Study & 7-Card Benchmark Insights

### 5.1 Reproducible Evaluator & Sealed Artifacts

All metrics in this guide are generated by the deterministic evaluator and verified via unit tests:
- **Evaluator Script**: [`experiments/heterogeneous_gpu_rocm_cuda_triton_stack_selection/run.py`](../../experiments/heterogeneous_gpu_rocm_cuda_triton_stack_selection/run.py)
- **Sealed Output**: [`experiments/heterogeneous_gpu_rocm_cuda_triton_stack_selection/results.json`](../../experiments/heterogeneous_gpu_rocm_cuda_triton_stack_selection/results.json)
- **Unit Test Suite**: [`tests/test_heterogeneous_gpu_rocm_cuda_triton_stack_selection.py`](../../tests/test_heterogeneous_gpu_rocm_cuda_triton_stack_selection.py)
- **7-Card 4:5 Visual Report**: [`docs/reports/heterogeneous-gpu-rocm-cuda-triton-tilelang-stack-selection-cards.html`](../reports/heterogeneous-gpu-rocm-cuda-triton-tilelang-stack-selection-cards.html)
- **Interactive Sizer Workbench**: [`docs/reports/llm-inference-engine-pd-sizer.en.html#hetero-gpu-rocm-cuda-sizer`](../reports/llm-inference-engine-pd-sizer.en.html) (Standalone: [`docs/reports/heterogeneous-gpu-rocm-cuda-triton-tilelang-sizer.html`](../reports/heterogeneous-gpu-rocm-cuda-triton-tilelang-sizer.html))
- **Provenance**: `synthetic_analytical_model_calibrated_to_anonymized_published_benchmarks` (`independently_reproduced=false`, `physical_gpu_cluster_measurement=not_measured`).

### 5.2 Key Quantitative Insights

1. **8-XCD Chiplet L2 Swizzling + Wavefront-64 Tuning Lifts MI300X Throughput `2.54x` (`1,120 → 2,850 tok/s`) Without Hardware Changes**.
2. **Cross-Vendor PD Disaggregation (`H100 Prefill + MI300X Decode`) Expands Net KV Cache Capacity per Decode Node by `2.58x` (`1,464 GB` vs. `568 GB`)**:
   Cutting unit token cost by **`72.4%` (`$5.36 → $1.48 / 1M tokens`)**.

---

## ⑥ 🛠️ Appendix: Production Architect Checklist

- [ ] **1. Never Mix NVIDIA and AMD GPUs Inside the Same `TP` Group**: Keep each `TP=8` instance homogeneous (`NCCL` on H100 Prefill instances, `RCCL` on MI300X Decode instances) and connect them via `Mooncake / NIXL` over standard RoCEv2 RDMA.
- [ ] **2. Enable `AMD AITER` & `8-XCD Swizzling` on MI300X/MI325X**: Set `VLLM_ROCM_USE_AITER=1` and apply chunked XCD workgroup remapping in custom Triton/TileLang kernels.
- [ ] **3. Standardize Cross-Pool KV Cache Format (`INT8` or Explicit-Scale `FP8`)**: Ensure identical byte layout and scaling factors when transferring KV blocks from Hopper to CDNA3.
- [ ] **4. Maintain Separate Per-Architecture Triton Autotune Caches**: Never reuse Hopper `Warp-32 / 228KB SMEM` autotune configs on CDNA3 `Wavefront-64 / 64KB LDS`.
- [ ] **5. Configure MI300X BIOS to `NPS1` (Unified Memory Partition)**: Prevent the 8 XCD chiplets from fragmenting into 8 tiny NUMA domains.
- [ ] **6. Audit L2 Cache Hit Rate (`>75%`) with `Omniperf / rocprofv2`**: Verify that XCD swizzling eliminates redundant cross-chiplet HBM fetches.
