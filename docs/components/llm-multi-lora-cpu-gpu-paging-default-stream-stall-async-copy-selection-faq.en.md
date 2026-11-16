# Multi-LoRA Serving CPU-to-GPU Adapter Paging (`max_loras` vs. `max_cpu_loras`) Default-Stream Decode Stall & `Stage-Before-Admit` Async Prefetch Selection Guide: Why Cold `r=64` LoRA Swaps Freeze `255` Active Decode Requests for `64.0 ms` (`5.5x` TPOT Spike) & How Pinned Host LRU + Background `lora_h2d_stream` Staging (`0.08 ms` Stall, `800x` Lower) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection/run.py` and `results.json`.

![Multi-LoRA CPU-to-GPU Adapter Paging & Stage-Before-Admit Visual Summary](../../assets/svg/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Three-Tier Pinned Adapter Hierarchy (`max_loras=8` HBM Slots + `max_cpu_loras=64` Pinned Host LRU + Async Disk Loader) + `Stage-Before-Admit` Background `lora_h2d_stream` Prefetching + In-Place Slot Reuse for `CUDAGraph` Preservation (`vLLM LRUCacheWorkerLoRAManager / SGLang / S-LoRA`)**.
- **Architect Value**: When serving `100–500` tenant LoRA adapters on a shared `70B` base model (`r = 64` across all 7 linear projections = **`320 MB` per adapter**), GPU HBM only holds a small active working set (`max_loras = 8` slots = `2.56 GB`). Naive adapter paging causes severe tail-latency spikes across all co-scheduled users whenever a cold LoRA request arrives:
  1. **The `64.00 ms` Head-of-Line Decode Freeze on Cold LoRA Admission (Tier 1)**:
     - Suppose 255 requests using warm GPU-resident LoRAs (`slots 0..6`) are actively decoding at `14.2 ms/token`.
     - Request #256 arrives requesting `LoRA #42` (a GPU cache miss).
     - If the scheduler immediately admits Request #256 into the running batch and loads `LoRA #42` from pageable CPU RAM onto the GPU on the default `compute_stream`, **all 255 active decode requests freeze for `64.00 ms`** (`5.0 GB/s` pageable staging bottleneck), spiking batch P99 TPOT from `14.20 ms` to **`78.20 ms` (`5.5x` spike)**!
  2. **The `11.50 ms` Pinned-Copy Compute-Stream Stall (Tier 2)**:
     - Even when `LoRA #42` is cached in Pinned Host RAM (`max_cpu_loras = 64`), enqueueing `cudaMemcpyAsync` on `compute_stream` (or synchronizing `compute_stream` on the copy before starting Layer 0 for the batch) still stalls all 255 active requests for **`11.50 ms`** while `320 MB` crosses PCIe!
  By **keeping cold-LoRA requests in a `STAGING_LORA` state while a dedicated `lora_h2d_stream` asynchronously DMAs the `320 MB` adapter from Pinned Host RAM into a pre-allocated GPU slot in parallel with the active batch's current decode step, and admitting the request only on the next scheduler tick once `cudaEventQuery(lora_ready_evt) == cudaSuccess` (`Stage-Before-Admit`, Tier 4)**, active decode interruption drops **`800.0x` (`64.00 ms -> 0.08 ms`)** and P99 TPOT remains rock-solid at **`14.28 ms`**!

---

## ① 📌 Executive Summary

1. **First-Principles Rule: Orthogonal PCIe Copy Engine vs. SM Compute Engine Decoupling (`Stage-Before-Admit`)**:
   - Because GPU PCIe Copy Engines (`CE`) operate independently of Streaming Multiprocessors (`SM`), a `320 MB` H2D transfer on `lora_h2d_stream` runs `100%` concurrently with a `14.2 ms` decode step on `compute_stream`—**provided `compute_stream` does not depend on the cold adapter during that step**!
   - Deferring the cold request's batch admission by 1 step (`Stage-Before-Admit`) hides `100%` of the PCIe transfer behind useful decode compute for the 255 warm requests!
2. **In-Place Slot Updates Preserve `CUDAGraph`s & Prevent Active-Set Thrashing**:
   - Copying adapter weights in-place (`gpu_lora_A[slot].copy_(pinned_cpu_A, non_blocking=True)`) into pre-allocated `[max_loras, ...]` HBM buffers keeps all base pointers constant for `CUDAGraph` replay, while pinning slots with `active_seq_count > 0` prevents round-robin thrashing when more than `max_loras` adapters are queued.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: How do we support heterogeneous LoRA ranks (`r = 8, 16, 32, 64`) inside a fixed-shape `CUDAGraph` slot pool?
- Allocate the GPU slot pool at `max_lora_rank = 64` (`gpu_lora_A[max_loras, L, 64, D_in]`). When loading an `r = 16` adapter into `slot`, copy its 16 rows into `:16` and pass the per-slot active rank `rank_table[slot] = 16` (or zero-pad `16:64`) to the `SGMV / BGMV` kernel.

### Q2: What happens if a requested LoRA misses both GPU (`max_loras=8`) and Pinned CPU (`max_cpu_loras=64`) caches?
- Offload disk/object-store (`S3 / NFS`) loading to a background thread pool that populates a Pinned CPU slot first, then triggers `Stage-Before-Admit` H2D DMA—never performing disk I/O on the scheduler event loop!

### Q3: How should waiting requests be ordered when 20 distinct LoRAs are queued and `max_loras = 8`?
- Group waiting requests by `lora_int_id` (**LoRA-Affinity Batching**) and only swap out a GPU slot when its active running sequence count drops to zero (`active_seq_count == 0`), amortizing each `320 MB` PCIe transfer across dozens of requests!

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Multi-LoRA Paging & Admission Scheduling Decision Tree

```mermaid
flowchart TD
    A["Multi-LoRA Server Receives Request R_256 for Cold LoRA #42 (320 MB, 255 Warm Reqs Decoding)"] --> B{"Where Are CPU-Tier LoRA Weights Stored?"}
    B -- "Pageable Host RAM or Disk" --> C["⛔ Tier 1: Synchronous staging on compute_stream freezes all 255 active decode requests for 64.00 ms!"]
    B -- "Pinned Host RAM Pool (max_cpu_loras=64)" --> D{"Which Stream Runs H2D Copy & When Is R_256 Admitted?"}
    D -- "Admit Immediately & Copy on compute_stream" --> E["⛔ Tier 2: 255 warm requests still wait 11.50 ms for PCIe transfer before Layer 0 starts!"]
    D -- "Async Copy Stream Without Active-Set Cap" --> F["⚠️ Tier 3: Thrashing when concurrent LoRAs > max_loras (2.40 ms stall)"]
    D -- "Stage-Before-Admit on lora_h2d_stream + cudaEventQuery + Affinity Grouping" --> G["🏆 Tier 4: 0.08 ms Decode Stall (800x lower) + 14.28 ms P99 TPOT + CUDAGraph Preserved!"]
```

### 3.2 Synchronous Compute-Stream Swap (`64.00 ms` Freeze) vs. `Stage-Before-Admit` Async DMA (`0.08 ms`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: Admit Before Staging (Freezes 255 Warm Requests for 64.00 ms!)"]
      B1["Step t: Admits cold R_256 into running batch immediately"] --> B2["compute_stream stalls 64.00 ms copying 320 MB LoRA #42!"]
      B2 --> B3["All 255 active requests suffer a 78.20 ms TPOT spike!"]
    end
    subgraph Good["🏆 Tier 4: Stage-Before-Admit on Dedicated lora_h2d_stream (0.08 ms Stall!)"]
      G1["Step t: Hold R_256 in STAGING; lora_h2d_stream DMAs 320 MB to slot #7 in background"] --> G2["compute_stream runs Step t Decode for 255 warm requests with 0 wait!"]
      G2 --> G3["Step t+1: EventQuery confirms slot #7 ready -> Admit R_256 with 0.08 ms overhead!"]
    end
```

### 3.3 Dual-Stream `Stage-Before-Admit` Prefetch & Admission Sequence

```mermaid
sequenceDiagram
    participant Sched as LoRA-Aware Scheduler
    participant H2D as Background lora_h2d_stream (PCIe DMA)
    participant Comp as Main compute_stream (255 Active Reqs)

    Sched->>Sched: 1. Step `t`: Req #256 needs cold `LoRA #42` -> allocate idle GPU `slot = 7`, set `STAGING_LORA`
    Sched->>H2D: 2. Launch `gpu_slot[7].copy_(pinned_cpu_lora[42], non_blocking=True)` + `record(evt[7])`
    Sched->>Comp: 3. Simultaneously launch Step `t` Decode for 255 warm requests (0.0 ms wait on `H2D`!)
    Note over H2D,Comp: PCIe Copy Engine transfers 320 MB in parallel while SMs execute Step `t` Decode!
    H2D-->>Sched: 4. Step `t+1`: `evt[7].query() == True` (0.08 ms) -> transition Req #256 to `RUNNING` on `slot = 7`!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Multi-LoRA Paging & Admission Policy | Active Decode Stall on Miss | Churn P99 `TPOT` | Effective PCIe BW | `CUDAGraph` Safe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Pageable Load on `compute_stream`** | Admits immediately; pageable copy | **`64.00 ms` ✕** | `78.20 ms` ✕ | `5.0 GB/s` ✕ | **No ✕** |
| **Tier 2: Pinned Copy on `compute_stream`** | Admits immediately; blocks Layer 0 | `11.50 ms` ✕ | `25.70 ms` ✕ | `27.8 GB/s` | Yes |
| **Tier 3: Layerwise Async (Unbounded Churn)** | Async stream, no active-set cap | `2.40 ms` | `16.60 ms` | `51.2 GB/s` | Yes |
| **Tier 4: Pinned LRU + `Stage-Before-Admit` (Recommended)** | **Background DMA before admission** | **`0.08 ms` (`800x` 🏆)** | **`14.28 ms` 🏆** | **`54.0 GB/s` 🏆** | **100% Safe 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection/run.py`](../../experiments/llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection/run.py) · Data: [`results.json`](../../experiments/llm_multi_lora_cpu_gpu_paging_default_stream_stall_async_copy_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-sizer.html`](../reports/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-selection-cards.html`](../reports/llm-multi-lora-cpu-gpu-paging-default-stream-stall-async-copy-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Pageable + Compute Stream", "decode_stall_ms": 64.00, "p99_tpot_ms": 78.20 },
      { "tier": "Tier 2: Pinned + Compute Stream", "decode_stall_ms": 11.50, "p99_tpot_ms": 25.70 },
      { "tier": "Tier 3: Layerwise Async (Unbounded)", "decode_stall_ms": 2.40, "p99_tpot_ms": 16.60 },
      { "tier": "Tier 4: Stage-Before-Admit + Affinity Group", "decode_stall_ms": 0.08, "p99_tpot_ms": 14.28 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "decode_stall_ms": "Quantity",
    "p99_tpot_ms": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Pinned Host LRU + Stage-Before-Admit Async DMA Cuts Cold-LoRA Decode Stall 800x (64ms -> 0.08ms)",
    "subtitle": "Active decode batch stall on cold r=64 LoRA miss (ms) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "decode_stall_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All multi-tenant LoRA serving clusters (`vLLM`, `SGLang`, `S-LoRA`) where total hosted adapters exceed `max_loras`.
- **⛔ Not for**: Copying cold LoRA weights on `compute_stream` or admitting cold-LoRA requests into the running batch before H2D DMA completes.

### Production Checklist
- [ ] **Pre-pin CPU-cached LoRA tensors (`max_cpu_loras`) using `cudaHostAlloc` / `.pin_memory()`**.
- [ ] **Enforce `Stage-Before-Admit` on a dedicated `lora_h2d_stream`**: Only transition a cold-LoRA request to `RUNNING` after `cudaEventQuery` confirms its GPU slot DMA is complete.
- [ ] **Update GPU LoRA slots via in-place `.copy_()` and pin slots with `active_seq_count > 0` to prevent thrashing**.
