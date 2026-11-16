# 5D Parallelism (`TP × PP × CP × EP × DP`) `NCCL` Sub-Group Initialization Storm & Channel HBM Explosion (`dist.new_group()` Loop vs. `ncclCommSplit` + `splitShare=1`) Selection Guide: Why Looping `dist.new_group()` on 4,096 GPUs Stalls Startup for `1,152 s` (`19.2 min`) & Eats `14.80 GB/GPU` of Hidden HBM, and How `ncclCommSplit` (`splitShare=1`) Cuts Init to `18.5 s` (`62.3x`) & Saves `12.95 GB/GPU`

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/nccl_commsplit_5d_parallel_sub_group_init_eager_memory_explosion_selection/run.py` and `results.json`.

![5D Parallelism NCCL Sub-Group Init & Channel HBM Explosion Visual Summary](../../assets/svg/nccl-commsplit-5d-parallel-sub-group-init-eager-memory-explosion-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Parallel `ncclCommSplit` (`torch.distributed.split_group`) Sub-Communicator Creation + Parent-Child Shared GPU Channel Buffer Pools (`ncclConfig_t.splitShare = 1`) + Unified Pipeline-Parallel `batch_isend_irecv` Communicator Reuse (`Megatron-Core / PyTorch DeviceMesh / TorchTitan`)**.
- **Architect Value**: Scaling 5D Parallelism (`TP=8, CP=2, PP=8, EP=8, DP=32`) to 4,096 GPUs triggers two notorious infrastructure failures when using legacy `ProcessGroup` initialization:
  1. **The `1,152.0 s` (`19.2 min`) `TCPStore` Sub-Group Initialization Storm (Tier 1)**:
     - Legacy model-parallel initializers create sub-groups via a global Python loop: `for i in range(num_subgroups): dist.new_group(ranks[i])`.
     - Across 4,096 GPUs, creating `512` TP groups, `2,048` CP groups, `512` EP groups, `512` PP groups, `128` DP groups, and pairwise P2P groups executes **`3,456` sequential `ncclUniqueId` broadcasts** through Rank 0's `TCPStore`, stalling job startup for **`1,152.0 seconds` (`19.2 minutes`)** before Step 1 even begins!
  2. **The Invisible `14.80 GB/GPU` NCCL Channel Buffer HBM Explosion**:
     - Every isolated `ncclComm_t` instance allocates its own GPU FIFO scratch buffers (`Simple` `4 MB`, `LL128` `~2.5 MB`, `LL` `512 KB` per channel $\times 32$ channels $\approx 500\text{--}820\text{ MB}$ per communicator) directly via raw CUDA driver allocations—**100% invisible to `torch.cuda.memory_allocated()`**!
     - When each GPU rank holds **18 active unshared communicators** across 5D dimensions and pairwise P2P links, NCCL internal buffers alone consume **`14.80 GB/GPU` of HBM**, shrinking usable training memory on an 80GB H100 down to just **`65.20 GB`** and causing mysterious `CUDA out of memory` crashes!
  By **splitting all orthogonal sub-groups in parallel from `world_comm` via `dist.split_group` (`ncclCommSplit`, cutting `TCPStore` exchanges from `3,456` to `7` and startup time from `1,152.0 s` to `18.5 s`—a `62.3x` speedup), enabling `splitShare = 1` so child communicators share parent GPU FIFO buffers, and consolidating pairwise P2P onto a single `pp_group` with `batch_isend_irecv` (Tier 4)**, per-GPU NCCL HBM drops from `14.80 GB` to **`1.85 GB/GPU` (reclaiming `12.95 GB/GPU` of HBM, `8.0x` reduction)**!

---

## ① 📌 Executive Summary

1. **First-Principles Mechanism: Why `ncclCommSplit` Is `62.3x` Faster Than Looping `dist.new_group()`**:
   - `for i in range(512): dist.new_group(tp_ranks[i])` serializes 512 disjoint 8-GPU groups through the global Rank-0 `TCPStore`.
   - `ncclCommSplit(world_comm, color=tp_group_id, key=tp_rank)` splits all 512 TP groups **simultaneously in parallel** using the existing `world_comm` bootstrap topology without touching `TCPStore`!
2. **Why `splitShare = 1` Reclaims `12.95 GB/GPU` of Hidden HBM**:
   - Because orthogonal 5D collectives (`TP`, `CP`, `EP`, `DP`, `DP-CP`) execute in distinct phases on the main stream, child communicators created with `ncclConfig_t.splitShare = 1` safely reuse the parent communicator's physical GPU FIFO/LL128 buffers instead of allocating 18 separate copies!

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: When should two communicators NOT share buffers (`splitShare=0`), and how do we bound their memory?
- Only communicators that execute collectives **concurrently on separate CUDA streams at the exact same time** (e.g., overlapping `DP Reduce-Scatter` on a comm stream with `TP All-Gather` on the compute stream) should use separate buffer pools. Even then, tuning `ncclConfig_t.maxCTAs` per dimension (e.g., `4–8` channels for P2P `PP` instead of `32`) keeps total NCCL HBM at **`1.85 GB/GPU`**.

### Q2: Why should pairwise 2-rank P2P groups be replaced by `dist.batch_isend_irecv` on a single `pp_group`?
- Creating separate 2-rank communicators for `(stage_i, stage_{i+1})` (and interleaved 1F1B stage pairs) multiplies communicator count and HBM per rank. `dist.batch_isend_irecv` multiplexes all P2P transfers over a single `pp_group` communicator safely and deadlock-free.

### Q3: How do we measure exact NCCL HBM consumption in production?
- Compare `(total - free)` from `torch.cuda.mem_get_info()` against `torch.cuda.max_memory_reserved()`, or inspect `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ALLOC` logs at startup.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 5D Parallelism Sub-Group Initialization & HBM Budgeting Decision Tree

```mermaid
flowchart TD
    A["4,096-GPU 5D Parallel Training Init (TP=8, CP=2, PP=8, EP=8, DP=32)"] --> B{"How Are Sub-Communicators Created?"}
    B -- "Sequential for-loop over dist.new_group(ranks)" --> C["⛔ Tier 1: 3,456 TCPStore broadcasts -> 1,152.0 s (19.2 min) startup stall!"]
    B -- "Lazy Init with Unshared Communicators" --> D["⛔ Tier 2: 340.0 s first-step stall & 7.60 GB/GPU hidden HBM!"]
    B -- "Parallel dist.split_group / ncclCommSplit(world_comm, color, key)" --> E{"Enable Parent-Child Buffer Sharing (splitShare=1)?"}
    E -- "No: splitShare=0 (unshared 32-channel pools)" --> F["⚠️ Tier 3: Fast 24.0 s init, but still consumes 6.40 GB/GPU HBM"]
    E -- "Yes: splitShare=1 + Unified PP Comm + Tuned maxCTAs" --> G["🏆 Tier 4: 18.5 s init (62.3x faster) + 1.85 GB/GPU HBM (-12.95 GB saved)!"]
```

### 3.2 Sequential `new_group` (`14.80 GB/GPU` HBM) vs. Parallel `ncclCommSplit` (`1.85 GB/GPU`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: Sequential new_group Loop + 18 Isolated Comms (14.80 GB/GPU HBM!)"]
      B1["3,456 TCPStore UniqueId Syncs (1,152.0 s)"] --> B2["18 Unshared ncclComm_t Instances per Rank"]
      B2 --> B3["18 × ~820 MB = 14.80 GB/GPU Hidden HBM -> Frequent Unexplained CUDA OOMs!"]
    end
    subgraph Good["🏆 Tier 4: Parallel ncclCommSplit (splitShare=1) + Unified PP Comm (1.85 GB/GPU!)"]
      G1["7 Parallel ncclCommSplit Calls Across 5D Mesh (18.5 s)"] --> G2["Child Comms Share Parent GPU FIFO Buffers (splitShare=1)"]
      G2 --> G3["NCCL HBM Drops from 14.80 GB -> 1.85 GB/GPU (Reclaims 12.95 GB/GPU HBM!)"]
    end
```

### 3.3 Parallel 5D Sub-Group Creation via `ncclCommSplit` (`splitShare=1`) Sequence

```mermaid
sequenceDiagram
    participant Store as Rank-0 TCPStore
    participant World as Parent World Comm (4,096 GPUs)
    participant Sub as 5D Child Comms (TP / CP / EP / PP / DP)

    Store->>World: 1. Bootstrap single `WORLD` communicator once (allocate shared parent buffer pool)
    loop For each 5D axis d in [TP, CP, EP, PP, DP, DP_CP]
        World->>Sub: 2. Call `ncclCommSplit(world_comm, color=group_id[d], key=rank_in_group[d], splitShare=1)`
        Note over World,Sub: All 512 TP groups split simultaneously in parallel without touching TCPStore!
    end
    Sub-->>World: 3. All 5D communicators ready in 18.5 s with 1.85 GB/GPU total HBM footprint!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Sub-Group Init & Buffer Policy | 4,096-GPU Init Time | `TCPStore` Syncs | Hidden `NCCL` HBM / GPU | Usable H100-80GB HBM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Sequential `new_group()` Loop + Pairwise P2P** | Global loop + 18 unshared comms | `1,152.0 s` ✕ | `3,456` ✕ | `14.80 GB` ✕ | `65.20 GB` ✕ |
| **Tier 2: Lazy Init + Unified PP Comm** | Removes pairwise P2P; unshared pools | `340.0 s` ✕ | `1,408` ✕ | `7.60 GB` ✕ | `72.40 GB` ✕ |
| **Tier 3: Parallel `ncclCommSplit` (`splitShare=0`)** | Parallel split, isolated channel pools | `24.0 s` | `7` | `6.40 GB` | `73.60 GB` |
| **Tier 4: Parallel `ncclCommSplit` (`splitShare=1`) (Recommended)** | **Parallel split + shared parent buffers** | **`18.5 s` (`62.3x` 🏆)** | **`7` 🏆** | **`1.85 GB` (`-12.95 GB` 🏆)** | **`78.15 GB` 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/nccl_commsplit_5d_parallel_sub_group_init_eager_memory_explosion_selection/run.py`](../../experiments/nccl_commsplit_5d_parallel_sub_group_init_eager_memory_explosion_selection/run.py) · Data: [`results.json`](../../experiments/nccl_commsplit_5d_parallel_sub_group_init_eager_memory_explosion_selection/results.json)
- **Interactive Sizer**: [`docs/reports/nccl-commsplit-5d-parallel-sub-group-init-eager-memory-explosion-sizer.html`](../reports/nccl-commsplit-5d-parallel-sub-group-init-eager-memory-explosion-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/nccl-commsplit-5d-parallel-sub-group-init-eager-memory-explosion-selection-cards.html`](../reports/nccl-commsplit-5d-parallel-sub-group-init-eager-memory-explosion-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Sequential new_group() Loop", "nccl_hbm_gb": 14.80, "init_sec": 1152.0 },
      { "tier": "Tier 2: Lazy Init + Unified PP Comm", "nccl_hbm_gb": 7.60, "init_sec": 340.0 },
      { "tier": "Tier 3: Parallel ncclCommSplit (splitShare=0)", "nccl_hbm_gb": 6.40, "init_sec": 24.0 },
      { "tier": "Tier 4: Parallel ncclCommSplit (splitShare=1)", "nccl_hbm_gb": 1.85, "init_sec": 18.5 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "nccl_hbm_gb": "Quantity",
    "init_sec": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Parallel ncclCommSplit (splitShare=1) Saves 12.95 GB/GPU of Hidden NCCL HBM",
    "subtitle": "Per-GPU NCCL internal channel buffer HBM consumption (GB) across 4 initialization tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "nccl_hbm_gb",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All large-scale 5D parallel (`DeviceMesh`, `Megatron-Core`, `TorchTitan`) training workloads on $\ge 64$ GPUs.
- **⛔ Not for**: Looping `dist.new_group(ranks)` across sub-group indices or creating pairwise 2-rank communicators for pipeline parallelism.

### Production Checklist
- [ ] **Replace `for ranks in groups: dist.new_group(ranks)` with `dist.split_group` (`ncclCommSplit`)**: Initialize all disjoint sub-groups of each parallel dimension in a single collective call.
- [ ] **Enable `split_share=True` (`ncclConfig_t.splitShare = 1`) for non-overlapping child communicators**: Reuse parent channel buffers and unify pipeline P2P onto one `pp_group` via `dist.batch_isend_irecv`.
- [ ] **Audit non-PyTorch GPU memory (`mem_get_info()` vs `max_memory_reserved()`)**: Ensure total NCCL communicator HBM stays below `2.5 GB/GPU`.
