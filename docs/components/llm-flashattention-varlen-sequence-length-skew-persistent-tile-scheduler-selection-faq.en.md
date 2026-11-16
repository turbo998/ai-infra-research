# `FlashAttention` Varlen (`cu_seqlens`) Sequence-Length Skew (`1× 64K + 63× 512`), Empty-CTA Wave Starvation & Persistent Dynamic Tile Scheduler (`Causal-Cost LPT + Split-KV`) Selection Guide: Why `flash_attn_varlen_func` Still Drops to `16.4%` SM Utilization (`18.50 ms`) on Skewed Batches & How Persistent `132-CTA` `LPT + Split-KV` Scheduling (`89.2%` SM, `3.40 ms`, `5.44x` Faster) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection/run.py` and `results.json`.

![FlashAttention Varlen Sequence-Length Skew & Persistent Tile Scheduler Visual Summary](../../assets/svg/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Persistent `132-CTA` Work-Stealing Kernel + Causal-Triangle Cost Longest-Processing-Time First (`Causal-Cost LPT`) Tile Ordering + Adaptive `Split-KV` Decomposition (`FlashAttention-3 Persistent Scheduler / FlashInfer Ragged Plan / Blackwell CLC`)**.
- **Architect Value**: Engineers often assume that replacing rectangular `padding=True` with 1D packed `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q)` eliminates all length-skew inefficiency. In reality, when a ragged batch combines **`1` long `65,536`-token (`64K`) sequence with `63` short `512`-token sequences**, traditional static-grid varlen kernels suffer a catastrophic **`5.44x` slowdown (`16.4%` SM utilization)**:
  1. **Static 3D CUDA Grid Sized to `max_seqlen_q = 65,536` (`98.4%` Empty CTA Launches, Tier 1)**: Because the host CPU does not inspect device-side `cu_seqlens` per sequence, a static kernel launches a rectangular 3D grid of `(ceil(65536 / 128), 64, H) = (512, 64, H)`. For each of the 63 short `512`-token sequences (which only need `4` Q-blocks), the hardware still launches `512` CTAs—`508` of which read `cu_seqlens`, see they are out-of-bounds, and immediately exit! Worse, once the short sequences finish in `0.2 ms`, **over `110` of the H100's `132` SMs sit 100% idle for `18.3 ms`** waiting for a handful of SMs to grind through the `64K` sequence (`16.4%` average SM utilization, `108.0 TFLOPS`)!
  2. **Unsorted FIFO Flattened Work Queue (`11.20 ms` Tail-Wave Straggler, Tier 2)**: Even when empty CTAs are removed, executing tiles in natural FIFO order schedules the heaviest bottom-of-triangle `64K` causal tiles last, leaving `> 100` SMs idle during the final wave!
  By deploying a **Persistent `132-CTA` Kernel (1 CTA per SM) pulling from a flattened work queue sorted by descending Causal KV Loop Cost (`LPT`: executing the heaviest bottom-of-triangle `64K` tiles first and using the `512`-token short tiles as fine-grained "sand" to pack the final wave) paired with Adaptive `Split-KV` (Tier 4)**, average SM utilization rises from `16.4%` to **`89.2%`**, effective throughput jumps from `108.0` to **`586.0 TFLOPS`**, and kernel latency drops **`5.44x` (`18.50 ms -> 3.40 ms`)**!

---

## ① 📌 Executive Summary

1. **First-Principles Load Balancing: Why Both Inter-Sequence Skew and Intra-Sequence Causal Skew Matter**:
   - **Inter-Sequence Skew**: A `64K` sequence has $\frac{65536^2}{512^2} = \mathbf{16,384\times}$ more attention FLOPs than a `512`-token sequence!
   - **Intra-Sequence Causal Triangle Skew**: Under `causal=True`, the first Q-tile of the `64K` sequence (rows `0..127`) iterates over only **`1` KV block**, whereas the last Q-tile (rows `65408..65535`) iterates over **`512` KV blocks (`512x` more compute)**!
2. **Causal-Cost `LPT` (Longest-Processing-Time First) Scheduling**:
   - In Graham's classicalmultiprocessor scheduling theory, sorting jobs in **descending order of processing time (`LPT`)** minimizes makespan:
     - Schedule the `64K` sequence's bottom-most Q-tiles (`512, 511, 510...` KV steps) **first** across all 132 SMs;
     - Schedule the 63 short sequences (`1–4` KV steps) **last** so they fill any remaining microsecond gaps across SMs at the end of the kernel!
3. **Adaptive `Split-KV` for Low-Head-Count GQA/MLA Long Sequences**:
   - When `num_heads` is small or a single `64K` sequence dominates the batch, splitting the deepest causal rows along the KV dimension across multiple SMs and merging via log-sum-exp (`LSE`) ensures all 132 SMs stay busy from $t = 0$ to completion.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Does atomic work-stealing (`atomicAdd(&tile_idx, 1)`) hurt L2 Cache KV locality?
- No, when paired with **L2-Aware Tile Swizzling**: group work tiles of similar causal cost by `(seq_id, kv_head_id)` so concurrent CTAs across the 132 SMs share the same KV pages in the 50 MB L2 cache while stepping through the `LPT` schedule.

### Q2: Why does `FlashInfer` require calling `wrapper.plan(indptr, ...)` before `wrapper.run()`?
- `plan()` inspects the sequence-length offsets once per step, splits long causal sequences into balanced virtual tiles (`Split-KV`), and writes a compact `LPT`-ordered tile descriptor table so the persistent kernel achieves `89%+` SM utilization without launching empty CTAs.

### Q3: How does `SM100 Blackwell` `Cluster Launch Control (CLC)` improve upon software `atomicAdd` persistent schedulers?
- `SM100 CLC` (`clusterlaunchcontrol.try_cancel`) steals unlaunched CTA clusters directly inside the hardware GPC scheduler, eliminating software queue management overhead while preserving dynamic load balancing.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Varlen Sequence-Length Skew Scheduler Decision Tree

```mermaid
flowchart TD
    A["Varlen Ragged Batch With Extreme Length Skew (1x 64K + 63x 512 Tokens)"] --> B{"CUDA Grid Launch & Tile Assignment Strategy?"}
    B -- "Static 3D grid sized to max_seqlen_q (64K)" --> C["⛔ Tier 1: 98.4% empty CTAs on short seqs + 83.6% SMs idle -> 18.50 ms!"]
    B -- "Flattened valid tiles in natural FIFO order" --> D["⛔ Tier 2: Heaviest 64K causal tiles run last -> 11.20 ms tail straggler!"]
    B -- "Persistent 132-CTA work-stealing kernel" --> E{"Causal-Cost LPT Ordering & Adaptive Split-KV Enabled?"}
    E -- "Q-tile stealing only (no causal LPT or Split-KV)" --> F["✅ Tier 3: 62.5% SM utilization, 4.85 ms latency (411 TFLOPS)"]
    E -- "Causal-Cost LPT (Heaviest First) + Short-Tile Tail Fill + Split-KV" --> G["🏆 Tier 4: 89.2% SM utilization + 586 TFLOPS + 3.40 ms (5.44x faster)!"]
```

### 3.2 Static `max_seqlen_q` Grid (`16.4%` SM) vs. Persistent `Causal-LPT + Split-KV` (`89.2%` SM)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1/2: Static 3D Grid or FIFO Queue (18.50 ms, 16.4% SM Utilization)"]
      B1["63 Short Seqs Finish in 0.2 ms"] --> B2["110+ SMs Sit Idle for 18.3 ms Waiting on 64K Causal Tail!"]
    end
    subgraph Good["🏆 Tier 4: Persistent 132-CTA + Causal LPT + Split-KV (3.40 ms, 89.2% SM)"]
      G1["Start: All 132 SMs Parallelize Heaviest 64K Bottom-of-Triangle Tiles"] --> G2["End: Fast 512-Token Tiles Fill Every SM Gap to Within 50 us!"]
    end
```

### 3.3 `Causal-Cost LPT` Planning & `132-CTA` Persistent Work-Stealing Sequence

```mermaid
sequenceDiagram
    participant Plan as Tile Planner (Causal LPT + Split-KV)
    participant Queue as Device Work-Tile Queue (Sorted Descending Cost)
    participant SMs as All 132 H100 SMs (Persistent CTAs)
    participant Reduce as Split-KV LSE Merge Epilogue

    Plan->>Queue: 1. Read `cu_seqlens`; split 64K deep causal rows & sort tiles heaviest-first
    loop Dynamic Work-Stealing Across 132 SMs
        SMs->>Queue: 2. Pull next tile via `tile_id = atomicAdd(&head, 1)`
        SMs->>SMs: 3. Execute heavy 64K causal tiles first across all 132 SMs
        SMs->>SMs: 4. Pack final wave gaps with lightweight 512-token tiles (89.2% SM util!)
    end
    SMs->>Reduce: 5. Merge Split-KV partial LSE outputs in 0.05 ms -> Complete in 3.40 ms!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Varlen Tile Scheduling Architecture | Kernel Latency | Average H100 SM Util | Effective `TFLOPS` | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Static 3D Grid (`max_seqlen_q`)** | Launches `512` blocks per short seq | `18.50 ms` ✕ | `16.4%` (Starved! ✕) | `108.0` ✕ | ⛔ **Severe Skew Slowdown** |
| **Tier 2: Unsorted FIFO Tile Queue** | Runs heavy `64K` causal rows last | `11.20 ms` ✕ | `27.1%` ✕ | `178.0` ✕ | ⛔ **Tail-Wave Stragglers** |
| **Tier 3: Persistent Q-Tile Stealing** | Steals Q-tiles without `Split-KV` | `4.85 ms` ✅ | `62.5%` ✅ | `411.0` ✅ | ✅ Moderate Length Variance |
| **Tier 4: Persistent `Causal-LPT + Split-KV` (Recommended)** | **Heaviest-first + sand fill + Split-KV** | **`3.40 ms` (`5.44x` Faster 🏆)** | **`89.2%` 🏆** | **`586.0` 🏆** | 🏆 **Ragged Serving & Packed SFT** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection/run.py`](../../experiments/llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection/run.py) · Data: [`results.json`](../../experiments/llm_flashattention_varlen_sequence_length_skew_persistent_tile_scheduler_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-sizer.html`](../reports/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-selection-cards.html`](../reports/llm-flashattention-varlen-sequence-length-skew-persistent-tile-scheduler-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Static 3D Grid (max_seqlen_q)", "latency_ms": 18.50, "sm_util_pct": 16.4 },
      { "tier": "Tier 2: Unsorted FIFO Flattened Queue", "latency_ms": 11.20, "sm_util_pct": 27.1 },
      { "tier": "Tier 3: Persistent Atomic Q-Tile Stealing", "latency_ms": 4.85, "sm_util_pct": 62.5 },
      { "tier": "Tier 4: Persistent Causal-LPT + Split-KV", "latency_ms": 3.40, "sm_util_pct": 89.2 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "latency_ms": "Quantity",
    "sm_util_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Persistent 132-CTA Causal-LPT + Split-KV Scheduler Speeds Up Skewed Varlen Attention 5.44x",
    "subtitle": "Kernel latency (ms) and average H100 SM utilization (%) on 1x 64K + 63x 512 varlen batch (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "latency_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All continuous-batching LLM engines (`vLLM / SGLang`), packed SFT/pretraining (`cu_seqlens`), and variable-resolution multimodal batches.
- **⛔ Not for**: Using static 3D CUDA grids sized to `max_seqlen_q` on batches with `10x–100x` sequence length skew.

### Production Checklist
- [ ] **Enable persistent work-stealing tile schedulers (`FlashAttention-3` / `FlashInfer Ragged Plan`)**: Launch exactly `num_SMs` persistent CTAs to eliminate empty out-of-bounds CTA waves.
- [ ] **Sort work tiles by descending causal KV loop cost (`Causal-Cost LPT`)**: Schedule the deepest bottom-of-triangle tiles first and pack the tail wave with short-sequence tiles.
- [ ] **Activate adaptive `Split-KV` on ultra-long sequences**: Partition long KV loops across SMs when a single long request dominates the batch.
