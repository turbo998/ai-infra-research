# Disaggregated Prefill-Decode (`PD`) Layerwise Pipelined KV Transfer & Contiguous Block Packing Selection Guide: Why End-of-Prefill Bulk Transfer + `16,000` Uncoalesced RDMA Writes Stalls First-Decode Handover for `148.0 ms` (`35.8%` NIC Util) & How Layerwise `cudaEvent` Pipelining + Coalesced Packing (`4.2 ms` Stall, `35.2x` Faster, `94.6%` NIC Util) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection/run.py` and `results.json`.

![Disaggregated PD Layerwise Pipelined KV Transfer Visual Summary](../../assets/svg/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Layerwise Event-Driven Asynchronous KV Push (`cudaEventRecord(layer_l)` Overlapping Layer $l$ RDMA Transfer with Layer $l+1$ Prefill Compute) + Double-Buffered Contiguous Block Packing (`16,000 -> 80` RDMA WQEs, `94.6%` 400GbE Line Rate) (`vLLM V1 KVConnector / SGLang Mooncake / NVIDIA NIXL`)**.
- **Architect Value**: In disaggregated Prefill-Decode (`PD`) serving, a `16,384`-token prompt on a `70B` (`80`-layer) model generates `2.68 GB` of KV cache that must be transferred from the Prefill node to the Decode node over RoCEv2 / InfiniBand. Naive KV transport implementations suffer two severe bottlenecks that create a jarring **`148.0 ms` freeze between Token 1 (emitted by Prefill) and Token 2 (emitted by Decode)**:
  1. **The `0.0%` Overlap End-of-Prefill Bulk Transfer Stall (Tier 1 & Tier 2)**:
     - Layer 0's KV cache is `100%` finished and immutable as soon as Layer 0's attention kernel completes (`~3 ms` into prefill).
     - Yet bulk transfer implementations leave the 400GbE RDMA NIC completely idle while the GPU computes Layers `0..79`, starting the `2.68 GB` transfer only *after* Layer 79 finishes—exposing `100%` of the network transfer latency on the user-facing critical path!
  2. **The `16,000`-WQE Scattered Paged-Block Doorbell Bottleneck (`35.8%` RNIC Utilization, Tier 1 & Tier 3)**:
     - Because PagedAttention allocates non-contiguous physical block IDs (`100` blocks for `16K` tokens) across `80` layers $\times 2$ (`K` and `V`), issuing a separate RDMA Write WQE for every `(layer, K/V, block_id)` tuple floods the RNIC with **`16,000` tiny work requests**—thrashing NIC doorbell/WQE fetch pipelines and collapsing 400GbE bandwidth utilization to **`35.8%`** (`148.0 ms` transfer time)!
  By **recording a `cudaEvent` immediately after each layer's KV cache write, gathering Layer $l$'s scattered blocks into a double-buffered contiguous GPU slab (`staging_buf[l % 2]`) on a dedicated transfer stream, and posting a single `33.5 MB` coalesced RDMA Write per layer while the GPU computes Layer $l+1$ (Tier 4)**, `92.8%` of the transfer is hidden behind prefill compute, RNIC utilization reaches **`94.6%`**, and exposed handover latency drops from `148.0 ms` to **`4.2 ms` (`35.2x` faster)**!

---

## ① 📌 Executive Summary

1. **First-Principles Law: Layerwise Compute-Transfer Overlap**:
   - Under Bulk End-of-Prefill transfer, exposed handover stall is $T_{\text{exposed}}^{\text{bulk}} = \sum_{l=0}^{L-1} t_{\text{xfer}}(l) = 148.0\text{ ms}$.
   - Under Layerwise Pipelined transfer, because layer prefill compute $t_{\text{comp}}(l+1)$ overlaps $t_{\text{xfer}}(l)$, Layers `0..78` arrive at the Decode node *before* Layer 79 finishes computing—leaving only the final layer's transfer + unpack exposed:
     $$T_{\text{exposed}}^{\text{pipe}} \approx t_{\text{xfer}}(L-1) + t_{\text{unpack}} = \mathbf{4.2\,\text{ms}} \quad (\mathbf{35.2\times}\text{ speedup})$$
2. **Why GPU Contiguous Gathering (`16,000 -> 80` WQEs) Boosts RNIC Efficiency from `35.8%` to `94.6%`**:
   - H100 HBM bandwidth (`3,350 GB/s`) is `67x` higher than 400GbE network bandwidth (`50 GB/s`). Spending `0.02 ms` on a GPU gather kernel to pack 100 scattered blocks into 1 contiguous buffer reduces RDMA WQE count by **`200x` (`16,000 -> 80`)**, saturating wire bandwidth at `94.6%`.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: How much extra HBM does the double-buffered contiguous staging slab (`staging_buf[2]`) require?
- Only **`8.4 MB` per GPU** (`67 MB` across an 8-GPU node)! Because the 2 staging slots are reused in a ring (`l % 2`) across all 80 layers, you only ever buffer 2 layers' worth of contiguous KV at a time.

### Q2: How does layerwise KV streaming work when `Chunked Prefill` splits a `16K` prompt into four `4K` chunks?
- As each `4K` chunk finishes its full 16-token blocks, those immutable blocks can be streamed layerwise immediately, or streamed during the final chunk's 80-layer forward pass once the Decode node allocates target pages.

### Q3: How are heterogeneous Tensor Parallel degrees (`TP_prefill = 8` -> `TP_decode = 4`) handled without an extra shuffle?
- Each Prefill rank $2d$ and $2d+1$ writes its contiguous single-KV-head slab directly to the left and right halves of Decode rank $d$'s 2-KV-head receive buffer via one-sided `RDMA Write`.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 Disaggregated PD KV Cache Transport Decision Tree

```mermaid
flowchart TD
    A["Disaggregated PD KV Transfer (70B 80 Layers, 16K Prompt = 2.68 GB KV, 400GbE RDMA)"] --> B{"When Does KV Transfer Start?"}
    B -- "After All 80 Layers Finish Prefill (Bulk End-of-Prefill)" --> C{"Are Scattered Paged Blocks Coalesced Before RDMA?"}
    C -- "No: Issue 16,000 per-block RDMA Write WQEs" --> D["⛔ Tier 1: 35.8% NIC utilization + 0% overlap -> 148.0 ms exposed handover stall!"]
    C -- "Yes: Gather into contiguous slab (80 WQEs)" --> E["⛔ Tier 2: 91.2% NIC util, but 0% overlap still exposes 58.4 ms stall!"]
    B -- "Layerwise Pipelined Push (Triggered via cudaEvent After Each Layer)" --> F{"Per-Layer Block Coalescing?"}
    F -- "No: 200 small WQEs per layer" --> G["⚠️ Tier 3: 74.3% overlap, but WQE overhead leaves 38.0 ms exposed"]
    F -- "Yes: Double-buffered GPU pack + 1 coalesced RDMA Write/layer" --> H["🏆 Tier 4: 92.8% Overlap + 94.6% NIC Util -> Only 4.2 ms Exposed Stall (35.2x faster)!"]
```

### 3.2 Bulk Uncoalesced Transfer (`148.0 ms` Stall) vs. Layerwise Coalesced Pipeline (`4.2 ms`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: End-of-Prefill Bulk + 16,000 Small WQEs (148.0 ms Exposed Stall!)"]
      B1["Prefill Layers 0..79: RNIC sits 100% idle!"] --> B2["Post-Prefill: Posts 16,000 tiny WQEs -> Decode stalls 148.0 ms before Token 2!"]
    end
    subgraph Good["🏆 Tier 4: Layerwise cudaEvent Pipeline + Double-Buffered Contiguous Pack (4.2 ms!)"]
      G1["Layer l finishes -> Pack into staging_buf[l%2] & post 1 RDMA Write while Layer l+1 computes!"] --> G2["When Layer 79 finishes, Layers 0..78 are already in Decode HBM -> 4.2 ms handover!"]
    end
```

### 3.3 Three-Stage Compute, Pack & RDMA Pipeline Sequence

```mermaid
sequenceDiagram
    participant Comp as Prefill Compute Stream
    participant Pack as GPU Pack Stream (staging_buf[2])
    participant RNIC as 400GbE GPUDirect RDMA NIC
    participant Dec as Decode Node HBM

    loop Layer l = 0 .. 79
        Comp->>Comp: 1. Run Layer `l` Attention & write Paged KV Cache
        Comp->>Pack: 2. `cudaEventRecord(evt[l])` -> `cudaStreamWaitEvent(pack_stream, evt[l])`
        Note over Comp,Pack: Compute stream immediately starts Layer `l` MLP & Layer `l+1` Attention!
        Pack->>RNIC: 3. Gather Layer `l` blocks into `staging_buf[l%2]` (0.02 ms) & post 1 RDMA Write
        RNIC->>Dec: 4. Stream Layer `l` KV directly into Decode staging slab at 94.6% line rate!
    end
    Dec->>Dec: 5. Layer 79 arrives -> scatter into Decode Paged Pool & launch Decode Step 1 (4.2 ms gap!)
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | PD KV Transport & Packing Policy | Exposed Handover Stall | 400GbE NIC Util | RDMA WQEs / Req | Compute Overlap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Bulk + 16,000 Small WQEs** | Waits for Layer 79; 1 WQE/block | **`148.0 ms` ✕** | `35.8%` ✕ | `16,000` ✕ | `0.0%` ✕ |
| **Tier 2: Bulk + Coalesced Slab** | Waits for Layer 79; 1 WQE/layer | `58.4 ms` ✕ | `91.2%` | `80` | `0.0%` ✕ |
| **Tier 3: Layerwise (Uncoalesced)** | Streams per layer; 200 WQEs/layer | `38.0 ms` | `42.5%` ✕ | `16,000` ✕ | `74.3%` |
| **Tier 4: Layerwise + Coalesced Pack (Recommended)** | **Streams per layer + 1 WQE/layer** | **`4.2 ms` (`35.2x` 🏆)** | **`94.6%` 🏆** | **`80` (`200x` 🏆)** | **`92.8%` 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection/run.py`](../../experiments/llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection/run.py) · Data: [`results.json`](../../experiments/llm_disaggregated_pd_layerwise_pipelined_kv_transfer_strided_block_packing_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-sizer.html`](../reports/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-selection-cards.html`](../reports/llm-disaggregated-pd-layerwise-pipelined-kv-transfer-strided-block-packing-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Bulk + 16,000 Small WQEs", "exposed_stall_ms": 148.0, "nic_util_pct": 35.8 },
      { "tier": "Tier 2: Bulk + Coalesced Slab", "exposed_stall_ms": 58.4, "nic_util_pct": 91.2 },
      { "tier": "Tier 3: Layerwise (Uncoalesced)", "exposed_stall_ms": 38.0, "nic_util_pct": 42.5 },
      { "tier": "Tier 4: Layerwise + Coalesced Pack", "exposed_stall_ms": 4.2, "nic_util_pct": 94.6 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "exposed_stall_ms": "Quantity",
    "nic_util_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Layerwise Pipelined KV Push + Coalesced Packing Cuts Exposed PD Handover Stall 35.2x (148ms -> 4.2ms)",
    "subtitle": "Exposed post-prefill KV transfer stall at 16K context (ms) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "exposed_stall_ms",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All disaggregated Prefill-Decode (`PD`) serving deployments (`vLLM`, `SGLang`, `Dynamo`) over RoCEv2 / InfiniBand.
- **⛔ Not for**: Waiting until Layer $L-1$ finishes before starting KV transfer or posting per-block RDMA WQEs.

### Production Checklist
- [ ] **Trigger asynchronous KV transfer after each layer via `cudaEventRecord(kv_ready_evt[l])`**.
- [ ] **Pack scattered PagedAttention blocks into a double-buffered contiguous staging slab (`1` RDMA Write WQE per layer)**.
- [ ] **Verify exposed `Token-1 -> Token-2` handover delay stays below `10 ms` at `16K` context**.
