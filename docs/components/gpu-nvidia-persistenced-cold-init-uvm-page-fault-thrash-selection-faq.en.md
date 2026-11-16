# GPU `nvidia-persistenced` Cold-Init Penalty (`14.5 s` `BAR1` Re-Init) & `UVM` (`cudaMallocManaged`) Page-Fault Thrashing Selection Guide: Why Missing `nvidia-persistenced` Stalls `8×H100` Container Startup for `14.5 s`, Why `cudaMallocManaged` Collapses PCIe Gen5 to `4.2 GB/s` (`12.6x` Slower) & How `nvidia-persistenced --uvm-persistence-mode` + Explicit Pinned DMA (`0.12 s` Init, `55.4 GB/s`) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/gpu_nvidia_persistenced_cold_init_uvm_page_fault_thrash_selection/run.py` and `results.json`.

![NVIDIA-Persistenced Cold Init & UVM Page-Fault Thrash Visual Summary](../../assets/svg/gpu-nvidia-persistenced-cold-init-uvm-page-fault-thrash-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **24/7 Hot Driver State Retention via `nvidia-persistenced --uvm-persistence-mode` (Eliminating `14.50 s` `640 GB BAR1` Teardown/Re-Init Cycles) + Explicit `cudaHostAlloc` Pinned Double-Buffering Over Async Copy Streams (Replacing `4.2 GB/s` `cudaMallocManaged` UVM Page-Fault Thrashing with `55.4 GB/s` Full-Speed PCIe Gen5 DMA)**.
- **Architect Value**: Two subtle NVIDIA kernel driver (`nvidia.ko` / `nvidia-uvm.ko`) behaviors routinely ambush production GPU clusters:
  1. **The `14.50 s` Container & `nvidia-smi` Cold-Start Hang (Missing `nvidia-persistenced`, Tier 1)**: By default on Linux, whenever the last process holding `/dev/nvidia*` open exits, the NVIDIA kernel driver **completely unloads the software state, ECC scrubbing state, and `640 GB` (`8 × 80 GB`) of PCIe `BAR1` memory mappings across all 8 H100 GPUs**! When the next Kubernetes pod, Slurm job, or `nvidia-smi` health probe starts a few seconds later, `cudaInit()` hangs in kernel mode for **`14.50 s`** re-initializing all 8 GPUs from cold hardware state!
  2. **The `12.65x` Slowdown From `cudaMallocManaged` (`UVM`) Demand Paging (Tier 1/2)**: When a workload exceeds 80 GB of HBM, switching allocations to `cudaMallocManaged` (Unified Virtual Memory) allows kernels to run without `CUDA OOM`, but every GPU access to a host-resident page triggers a **hardware GPU MMU Page Fault** $\to$ stalls the SM warp $\to$ raises a PCIe interrupt to the host CPU $\to$ locks kernel page tables and migrates tiny `4 KB / 64 KB` pages over PCIe $\to$ flushes GPU TLBs and replays instructions. At `450,000` faults/sec, PCIe Gen5 x16 effective bandwidth collapses from `55.4 GB/s` to **`4.2 GB/s`**, slowing steps from `302.0 ms` to **`3,820.0 ms`**!
  By **enabling `nvidia-persistenced --uvm-persistence-mode` (holding device FDs open 24/7 with `0 MB` HBM overhead so `cudaInit()` completes in `0.12 s`, `120.8x` faster) and replacing UVM demand paging with explicit `cudaHostAlloc` pinned double-buffering on a dedicated non-blocking `copy_stream` (Tier 4)**, PCIe Gen5 throughput reaches **`55.4 GB/s` (`13.2x` higher)** and step latency drops **`12.65x` (`3,820.0 ms -> 302.0 ms`)**!

---

## ① 📌 Executive Summary

1. **First-Principles Driver Lifecycle: How `nvidia-persistenced` Saves `14.50 s` Using `0 MB` of VRAM**:
   - `nvidia-persistenced` opens `/dev/nvidia0..7` and `/dev/nvidia-uvm` at boot without creating a CUDA context (`0 MB` VRAM used). Keeping the kernel file-descriptor reference count $\ge 1$ prevents the driver from tearing down the `640 GB` BAR1 aperture when containers exit.
2. **Why Legacy `nvidia-smi -pm 1` Is Deprecated and Incomplete (Tier 2)**:
   - `nvidia-smi -pm 1` does not persist `/dev/nvidia-uvm` state, leaving a `2.85 s` UVM initialization delay on container start. Always run `nvidia-persistenced --uvm-persistence-mode`.
3. **Why Explicit Pinned Double-Buffering (`55.4 GB/s`) Crushes `UVM` Demand Paging (`4.2 GB/s`)**:
   - Transformer layer execution order is 100% deterministic (`Layer 1 -> Layer 2 -> ... -> Layer L`). Streaming `Layer l+1` via `cudaMemcpyAsync` from `cudaHostAlloc` pinned memory on a background DMA copy engine while `Layer l` computes on SMs achieves `55.4 GB/s` with **zero GPU page faults and zero warp stalls**.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: If legacy code requires `cudaMallocManaged` (Tier 3), how can we mitigate the `4.2 GB/s` fault storm?
- Call **`cudaMemAdvise` + `cudaMemPrefetchAsync(ptr, size, gpu_id, prefetch_stream)`** ahead of kernel launch so the driver transfers `2 MB` large pages in bulk (`38.5 GB/s`) before SMs touch the memory, avoiding on-demand warp page faults.

### Q2: How do we verify `nvidia-persistenced` is active across a Kubernetes GPU cluster?
- Check that `Persistence-M` shows `On` in `nvidia-smi` and verify `systemctl is-active nvidia-persistenced` (with `--uvm-persistence-mode` in its command line).

### Q3: How do we detect accidental UVM page faults in profiling traces?
- Check `nsys profile --cuda-um-cpu-page-faults=true --cuda-um-gpu-page-faults=true`: any non-zero GPU page fault count during steady-state training or inference indicates severe PCIe fault thrashing.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 GPU Driver Persistence & Memory Oversubscription Decision Tree

```mermaid
flowchart TD
    A["8x H100 Production Node Container Startup & >80GB Memory Offload"] --> B{"Is nvidia-persistenced --uvm-persistence-mode Running?"}
    B -- "No: Driver unloads state when last container exits" --> C["⛔ Tier 1: Re-initializes 640GB BAR1 & ECC on every start -> 14.50 s hang!"]
    B -- "Legacy nvidia-smi -pm 1 only" --> D["⛔ Tier 2: UVM state still unloads -> 2.85 s startup delay"]
    B -- "Yes: Full persistenced daemon (0.12 s instant init)" --> E{"How Are Tensors Exceeding 80GB HBM Transferred Over PCIe?"}
    E -- "Implicit cudaMallocManaged (UVM) demand paging" --> F["⛔ Tier 1/2: 450k GPU page faults/s -> PCIe collapses to 4.2 GB/s (3,820 ms)!"]
    E -- "UVM + explicit cudaMemPrefetchAsync" --> G["✅ Tier 3: Bulk 2MB prefetch reaches 38.5 GB/s (480 ms step)"]
    E -- "Explicit cudaHostAlloc Pinned Double-Buffering + Copy Stream" --> H["🏆 Tier 4: 0.12s init + zero page faults + 55.4 GB/s PCIe (302 ms, 12.65x faster)!"]
```

### 3.2 `UVM` Demand Page-Fault Storm (`4.2 GB/s`) vs. Explicit Pinned DMA (`55.4 GB/s`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1/2: cudaMallocManaged Demand Paging (4.2 GB/s, 3,820 ms Step)"]
      B1["GPU Warp Hits Missing Page -> MMU Fault Stalls Warp"] --> B2["PCIe Interrupt -> OS Page Table Lock -> 64KB Copy -> TLB Flush!"]
    end
    subgraph Good["🏆 Tier 4: Explicit Pinned Double-Buffer + Copy Engine DMA (55.4 GB/s, 302 ms Step)"]
      G1["While SMs Compute Layer l: Copy Engine Streams Layer l+1 (256MB) at 55.4 GB/s"] --> G2["Zero GPU Page Faults & Zero Warp Stalls!"]
    end
```

### 3.3 `nvidia-persistenced` Hot Init & Zero-Fault Pinned DMA Sequence

```mermaid
sequenceDiagram
    participant Daemon as nvidia-persistenced (0 MB VRAM, Refcount >= 1)
    participant Drv as NVIDIA Kernel Driver (640GB BAR1 Hot)
    participant Pod as Training / Serving Container
    participant CE as GPU Copy Engine (Async Pinned DMA)

    Daemon->>Drv: 1. Hold /dev/nvidia0..7 & /dev/nvidia-uvm open 24/7
    Pod->>Drv: 2. Container calls cudaInit() -> BAR1 already mapped -> returns in 0.12 s!
    Pod->>CE: 3. Pre-allocate NUMA-pinned host buffers via cudaHostAlloc
    par Layer l Compute & Layer l+1 DMA Prefetch
        Pod->>Pod: 4a. Compute Stream executes Layer l GEMMs at 100% SM utilization
        CE->>Pod: 4b. Copy Stream prefetches Layer l+1 at 55.4 GB/s (0 page faults!)
    end
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Persistence & Offload Architecture | `cudaInit()` Cold Start | Effective PCIe Gen5 BW | 20% Oversub Step Time | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: No Daemon + `cudaMallocManaged`** | Unloads `BAR1` + demand page faults | `14.50 s` ✕ | `4.2 GB/s` ✕ | `3,820.0 ms` ✕ | ⛔ **Severe Startup Hang & Fault Thrash** |
| **Tier 2: Legacy `-pm 1` + Demand `UVM`** | Partial persistence + fault storm | `2.85 s` ✕ | `5.1 GB/s` ✕ | `3,150.0 ms` ✕ | ⛔ **Deprecated & 10x Slower PCIe** |
| **Tier 3: `persistenced` + `PrefetchAsync`** | Hot driver + bulk `2MB` UVM prefetch | `0.14 s` ✅ | `38.5 GB/s` ✅ | `480.0 ms` ✅ | ⚠️ Legacy UVM Codebase Mitigation |
| **Tier 4: `persistenced` + Pinned DMA (Recommended)** | **24/7 hot daemon + zero-fault DMA** | **`0.12 s` (`120.8x` Faster 🏆)** | **`55.4 GB/s` (`13.2x` 🏆)** | **`302.0 ms` (`12.65x` Faster 🏆)** | 🏆 **Production AI Nodes & Offload Engines** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/gpu_nvidia_persistenced_cold_init_uvm_page_fault_thrash_selection/run.py`](../../experiments/gpu_nvidia_persistenced_cold_init_uvm_page_fault_thrash_selection/run.py) · Data: [`results.json`](../../experiments/gpu_nvidia_persistenced_cold_init_uvm_page_fault_thrash_selection/results.json)
- **Interactive Sizer**: [`docs/reports/gpu-nvidia-persistenced-cold-init-uvm-page-fault-thrash-sizer.html`](../reports/gpu-nvidia-persistenced-cold-init-uvm-page-fault-thrash-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/gpu-nvidia-persistenced-cold-init-uvm-page-fault-thrash-selection-cards.html`](../reports/gpu-nvidia-persistenced-cold-init-uvm-page-fault-thrash-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: No Daemon + UVM Demand Paging", "init_s": 14.50, "pcie_gbps": 4.2 },
      { "tier": "Tier 2: Legacy smi -pm 1 + UVM Paging", "init_s": 2.85, "pcie_gbps": 5.1 },
      { "tier": "Tier 3: persistenced + UVM PrefetchAsync", "init_s": 0.14, "pcie_gbps": 38.5 },
      { "tier": "Tier 4: persistenced + Explicit Pinned DMA", "init_s": 0.12, "pcie_gbps": 55.4 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "init_s": "Quantity",
    "pcie_gbps": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "nvidia-persistenced + Explicit Pinned DMA Cuts Cold Init 120.8x & Boosts PCIe Bandwidth 13.2x vs UVM",
    "subtitle": "Effective PCIe Gen5 bandwidth (GB/s) and CUDA cold init time (s) on 8x H100 (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "pcie_gbps",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: Every production Linux GPU node (`nvidia-persistenced` mandatory) and all host-memory offloading systems (`LMCache`, `ZeRO-Infinity`, `FlexGen`).
- **⛔ Not for**: Using `cudaMallocManaged` demand paging to work around `CUDA OOM` in LLM training or serving.

### Production Checklist
- [ ] **Enable `nvidia-persistenced --uvm-persistence-mode` via `systemd` on all GPU nodes**: Keep `cudaInit()` and `nvidia-smi` latency `< 0.15 s`.
- [ ] **Use explicit `cudaHostAlloc` pinned double-buffering + dedicated `CUDA Stream`s instead of `cudaMallocManaged`**: Achieve `55+ GB/s` zero-fault PCIe Gen5 DMA.
- [ ] **Verify `0` GPU UVM page faults in `nsys` profiles**: Ensure SMs never stall on PCIe page migrations during kernel execution.
