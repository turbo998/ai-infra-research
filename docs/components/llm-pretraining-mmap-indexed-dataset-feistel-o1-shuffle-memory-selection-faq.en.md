# 10T-Token Pretraining Memory-Mapped Dataset (`Megatron IndexedDataset .bin/.idx`) & `5-Billion` Sample Global Shuffle (`np.random.permutation` Host OOM vs. `O(1)` `Feistel` Bijective Cipher) Selection Guide: Why `np.random.permutation(5B)` Consumes `320 GB/Node` (`OOMKilled`), Why Unmanaged `mmap` Stalls `kswapd` & How 4-Round `Cycle-Walking Feistel` + `MADV_DONTNEED` (`0.001 GB RAM`, `0.08 s` Init) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection/run.py` and `results.json`.

![10T-Token Pretraining mmap IndexedDataset & O(1) Feistel Shuffle Visual Summary](../../assets/svg/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **`O(1)` Register-Space Global Bijective Shuffling via 4-Round `Cycle-Walking Feistel Networks` + Read-Once `mmap` Page Reclamation (`posix_madvise(MADV_DONTNEED)`) for Multi-Trillion-Token Pretraining (`Megatron-Core / MosaicML Streaming / LitData / TorchTitan`)**.
- **Architect Value**: Scaling LLM pretraining corpora to **`10T–15T` tokens ($N = 5,000,000,000$ sequences of `2,048` tokens)** breaks standard PyTorch `Sampler` and `mmap` patterns on 8-GPU nodes:
  1. **Per-Rank `np.random.permutation(5_000,000,000)` Host Memory Explosion (`320.0 GB/Node`, Tier 1)**: A single `5B`-element `int64` index array requires **`40.0 GB` of RAM**. When all 8 local GPU ranks generate or load their own copy in Python, they consume **`320.0 GB` of host RAM per node** (plus `142.0 s` of RNG initialization and `DataLoader` fork Copy-on-Write duplication), immediately triggering container `OOMKilled`!
  2. **Unmanaged Random `mmap` Reads Over `20 TB` `.bin` Files (Tier 2)**: Even if the index array is shared via `mmap`, random reads across `20 TB` of memory-mapped `.bin` token files fill 100% of host RAM with single-use Linux Page Cache pages, triggering continuous `kswapd` page-reclaim stalls during training!
  By replacing materialized index arrays with a **4-Round `Cycle-Walking Feistel Cipher` (computing a collision-free bijective mapping $i \mapsto \pi_{\text{seed}}(i) \in [0, N)$ on the fly in `20 ns` using $\mathcal{O}(1)$ registers) and calling `madvise(MADV_DONTNEED)` after reading each `mmap` slice (Tier 4)**, node shuffle index memory drops from `320.0 GB` to **`0.001 GB`**, and epoch initialization / mid-epoch checkpoint resume drops **`1,775x` (`142.0 s -> 0.08 s`)**!

---

## ① 📌 Executive Summary

1. **First-Principles `O(1)` Global Shuffling (`Cycle-Walking Feistel Network`)**:
   - Shuffling $N$ items is mathematically equivalent to evaluating a seeded **bijective permutation** $\pi_{\text{seed}} : \{0, \dots, N-1\} \to \{0, \dots, N-1\}$.
   - A 4-round balanced Feistel network on $2k$ bits (where $2^{2k} \ge N$, e.g. $2k = 34$ bits for $N = 5 \times 10^9$) is strictly invertible: every input maps to a unique output. Whenever the output exceeds $N-1$, feeding it back into the cipher (**Cycle-Walking**) follows the permutation cycle until it lands in $[0, N)$ (taking $< 2$ iterations on average).
   - **Result**: **`0 Bytes` of index array memory**, **`0.08 s` startup**, and $\mathcal{O}(1)$ instant seek to any training step $s$ on checkpoint resume!
2. **Preventing Silent Token ID Truncation on Large Vocabularies (`uint32` vs `uint16`)**:
   - Legacy `Megatron IndexedDataset` defaulted to `uint16` (`0..65535`). For modern tokenizers (`Llama-3 128K`, `Qwen 152K`, `Gemma 256K`), always verify `.bin` files are serialized as **`uint32`** to prevent silent modulo-65536 wrap-around corruption.
3. **Keeping Host Page Cache Flat with `MADV_DONTNEED`**:
   - Calling `madvise(page_aligned_addr, length, MADV_DONTNEED)` after copying a batch from the `.bin` `mmap` immediately releases physical page frames back to the Linux kernel.

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: Why is `O(1)` global `Feistel` shuffling superior to a `100K`-sample streaming reservoir buffer (`IterableDataset`, Tier 3)?
- A `100K` sliding buffer only shuffles within a narrow `~200M`-token local window. If the underlying `.bin` files concatenate large domain dumps (e.g., GitHub code followed by Wikipedia followed by ArXiv), a local buffer causes severe **domain clustering and loss spikes**, whereas `Feistel` mixes samples uniformly across the entire `10T`-token corpus.

### Q2: How do we avoid tiny random `8 KB` disk seeks on Lustre/3FS when using global `Feistel` permutation?
- Use **Two-Level Hierarchical Feistel Shuffling**: apply an outer `Feistel` permutation across `1 MB` contiguous macro-blocks (`64` sequences) to preserve high sequential storage read bandwidth, and an inner `Feistel` permutation across sequences within the prefetched macro-bricks in RAM.

### Q3: How should static metadata arrays (like document byte offsets in `.idx`) be shared across 8 local ranks on a node?
- Never load them with `pickle.load()` or `np.load()` per rank; map them read-only via `np.memmap("dataset.idx", mode="r")` so all 8 processes share a single physical OS page-cache copy.

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 10T-Token Pretraining Dataset Indexing & Global Shuffle Decision Tree

```mermaid
flowchart TD
    A["10T-Token Pretraining Corpus (N = 5 Billion Sequences, 8 Ranks/Node)"] --> B{"How Is Global Epoch Shuffling Implemented?"}
    B -- "Per-rank np.random.permutation(5B) int64 array in RAM" --> C["⛔ Tier 1: 40 GB × 8 = 320 GB/node! Instant container OOMKilled & 142s stall!"]
    B -- "Shared mmap .idx file without MADV_DONTNEED" --> D["⛔ Tier 2: Random reads over 20TB .bin exhaust Linux Page Cache & stall kswapd!"]
    B -- "Zero-Array O(1) Index Space" --> E{"Local Window Shuffle vs. True Global Bijective Shuffle?"}
    E -- "100K-sample streaming reservoir buffer" --> F["⚠️ Tier 3: 2.4 GB RAM, but suffers domain clustering across large shards"]
    E -- "4-Round Cycle-Walking Feistel Cipher + mmap MADV_DONTNEED" --> G["🏆 Tier 4: 0.001 GB RAM (saves 320GB!) + 0.08s instant init/resume + zero kswapd!"]
```

### 3.2 `np.random.permutation(5B)` (`320 GB` OOM) vs. `O(1)` `Feistel` Bijective Shuffle (`0.001 GB`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1: 8 Ranks Allocating 5B int64 Arrays (320.0 GB RAM -> OOM!)"]
      B1["Ranks 0..7 Call np.random.permutation(5,000,000,000)"] --> B2["Allocates 320 GB Host RAM + 142.0 s Init -> Host OOMKilled!"]
    end
    subgraph Good["🏆 Tier 4: 4-Round Cycle-Walking Feistel Permutation (0.001 GB RAM, 0.08 s)"]
      G1["Input Logical Step Index i & Epoch Seed"] --> G2["Computes Bijective Target Sample Index in 20 ns (Zero Array Allocation)!"]
    end
```

### 3.3 `O(1)` `Feistel` Lookup & `MADV_DONTNEED` Page Reclamation Sequence

```mermaid
sequenceDiagram
    participant Train as Training Loop (Step s, Rank r)
    participant Feist as Cycle-Walking Feistel Permuter (O(1) Registers)
    participant Mmap as Read-Only mmap (.bin / .idx)
    participant Kernel as Linux VMM Page Cache

    Train->>Feist: 1. Query sample index for logical position i = s * global_batch + r
    Feist-->>Train: 2. Return bijective shuffled_idx = feistel_permute(i, N=5B, seed) in 20 ns
    Train->>Mmap: 3. Read 2,048-token uint32 slice from mmap .bin
    Mmap->>Kernel: 4. Load 8KB page & copy to pinned host batch buffer
    Train->>Kernel: 5. Call madvise(MADV_DONTNEED) -> immediately reclaim single-use page frame!
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | `5B`-Sample Shuffle & `mmap` Policy | Node Shuffle Index RAM | Epoch Init / Resume Time | Page-Cache Thrashing | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Per-Rank `np.random.permutation(5B)`** | 8 processes allocate `40 GB` arrays | `320.0 GB` (OOM! ✕) | `142.0 s` ✕ | Severe ✕ | ⛔ **Guaranteed Host OOM at Scale** |
| **Tier 2: Shared `mmap` `.idx` (No `madvise`)** | Shared index, unmanaged `.bin` cache | `64.0 GB` ✕ | `48.5 s` ✕ | High (`kswapd`! ✕) | ⛔ **Periodic I/O Stalls After 1 Hour** |
| **Tier 3: Local `100K` Window Buffer** | Reservoir sampling in narrow window | `2.4 GB` ✅ | `1.20 s` ✅ | None ✅ | ⚠️ Pre-Shuffled Shards Only |
| **Tier 4: `O(1)` `Feistel` + `MADV_DONTNEED` (Recommended)** | **Bijective cipher + page reclamation** | **`0.001 GB` (`-320 GB` 🏆)** | **`0.08 s` (`1,775x` Faster 🏆)** | **Zero Thrash 🏆** | 🏆 **Multi-Trillion-Token Pretraining** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection/run.py`](../../experiments/llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection/run.py) · Data: [`results.json`](../../experiments/llm_pretraining_mmap_indexed_dataset_feistel_o1_shuffle_memory_selection/results.json)
- **Interactive Sizer**: [`docs/reports/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-sizer.html`](../reports/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-selection-cards.html`](../reports/llm-pretraining-mmap-indexed-dataset-feistel-o1-shuffle-memory-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Per-Rank np.random.permutation(5B)", "ram_gb": 320.0, "init_s": 142.0 },
      { "tier": "Tier 2: Shared mmap .idx (No madvise)", "ram_gb": 64.0, "init_s": 48.5 },
      { "tier": "Tier 3: Local 100K Window Buffer", "ram_gb": 2.4, "init_s": 1.2 },
      { "tier": "Tier 4: O(1) Feistel Cipher + MADV_DONTNEED", "ram_gb": 0.001, "init_s": 0.08 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "ram_gb": "Quantity",
    "init_s": "Quantity"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "O(1) Cycle-Walking Feistel Permutation Eliminates 320 GB/Node Shuffle Index Memory & Starts in 0.08s",
    "subtitle": "Host RAM consumed by shuffle indices per 8-GPU node (GB) for 5B sequences (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "ram_gb",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All large-scale LLM pretraining and continual pretraining pipelines ($N \ge 10^7$ sequences).
- **⛔ Not for**: Calling `np.random.permutation(N)` or `torch.randperm(N)` in memory on billion-sample corpora.

### Production Checklist
- [ ] **Replace materialized index arrays with a 4-round `Cycle-Walking Feistel` bijective function**: Achieve global shuffling and instant mid-epoch seek in $\mathcal{O}(1)$ RAM.
- [ ] **Enforce `uint32` token storage for vocabularies $\ge 65,536$**: Prevent silent `uint16` overflow corruption.
- [ ] **Call `madvise(MADV_DONTNEED)` on consumed `mmap` slices**: Prevent 20 TB `.bin` files from exhausting host Linux Page Cache.
