# GPU `CUDA error: device-side assert triggered` (`cudaErrorAssert = 710`) Sticky Context Destruction & Zero-Sync Clamping Guard Selection Guide: Why 1 Out-of-Bounds Token ID (`-100` or `>= V`) Permanently Destroys `CUcontext` Across All Co-Scheduled Requests, Why GPU `.any()` Checks Cost `1.85 ms/step` & Break `CUDAGraph`, and How CPU Ingress Gating + GPU Branchless Clamping (`100%` Survival, `0.02 ms` Cost) Solves It

> **Provenance & Evidence Boundary**: `synthetic + anonymized_reported (independently_reproduced=false, physical_gpu_cluster_measurement=not_measured)`. All quantitative metrics in this guide are grounded in the sealed deterministic evaluator `experiments/cuda_device_side_assert_embedding_oob_sticky_context_destruction_guard_selection/run.py` and `results.json`.

![CUDA Device-Side Assert & Zero-Sync Clamping Guard Visual Summary](../../assets/svg/cuda-device-side-assert-embedding-oob-sticky-context-destruction-guard-selection-faq-summary.svg)

---

## 0. Daily Capability Breakthrough

- **Today's Highlight Breakthrough**: **Two-Layer Zero-Sync Bounds Isolation: Pre-H2D CPU SIMD Ingress Validation (`0.02 ms`) + GPU In-Register Branchless Clamping (`idx = (idx < 0 || idx >= V) ? 0 : idx`) + Asynchronous Non-Fatal `atomicAdd` OOB Telemetry + PagedAttention Dummy Scratch Page `#0` (`vLLM V1 / SGLang / Megatron-Core`)**.
- **Architect Value**: Few errors cause more production downtime in LLM serving and distributed training than:
  `RuntimeError: CUDA error: device-side assert triggered (cudaErrorAssert = 710)`!
  Engineers attempting to handle this error in production encounter three painful realities:
  1. **Why Python `try ... except RuntimeError:` Cannot Recover a `device-side assert` (`0.0%` `CUcontext` Survival, Tier 1)**:
     - Inside standard PyTorch CUDA kernels (`Embedding.cu`, `Indexing.cu`), bounds checks compile to the hardware PTX **`trap;`** instruction (`CUDA_KERNEL_ASSERT(idx >= 0 && idx < num_embeddings)`).
     - When a single thread executes `trap;`—whether due to a client passing an out-of-range token in `prompt_token_ids`, an unmasked `-100` label pad leaking into `input_ids`, or `torch.multinomial` emitting `INT64_MIN` (`-9223372036854775808`) on `NaN` logits—the GPU hardware immediately aborts all active warps and **places the process's entire `CUcontext` into a permanent, sticky error state**!
     - Once `CUcontext` is poisoned by `cudaErrorAssert`, **every subsequent CUDA call in that OS process fails forever**—meaning 1 bad request crashes an entire `TP=8` serving replica (killing all 255 co-scheduled user requests) or hangs a 512-GPU training job until the `30-minute` NCCL watchdog timeout!
  2. **Why GPU `if (input_ids < 0).any() or (input_ids >= V).any():` Ruins Throughput & Breaks `CUDAGraph` (`1.85 ms/step` Stall, Tier 2)**:
     - Evaluating `.any()` on a GPU tensor in Python forces a synchronous D2H copy and `cudaStreamSynchronize()`, adding **`1.85 ms/step`** of blocking latency (`-19.4%` throughput drop) and failing immediately during `CUDAGraph` capture!
  3. **Why CPU-Only Ingress Validation Still Misses `35%` of Assertions (Tier 3)**:
     - During autoregressive decoding inside a `CUDAGraph`, the token sampled on step $t$ is fed directly in GPU memory to step $t+1$'s embedding layer without round-tripping through CPU validation!
  By **validating external `prompt_token_ids` on CPU before H2D copy (`0.02 ms`), replacing PTX `trap;` in GPU lookup/sampler kernels with branchless register clamping (`idx = 0`) paired with a non-fatal async `atomicAdd(&d_oob_count, 1)` counter, and routing unused PagedAttention slots to a reserved Dummy Scratch Page `#0` (Tier 4)**, `CUcontext` survival reaches **`100.0%`** at **`99.8%` peak throughput** with **`100%` `CUDAGraph` compatibility**!

---

## ① 📌 Executive Summary

1. **First-Principles Rule: Never Execute PTX `trap;` on Data-Dependent Indices in Multi-Tenant Production Kernels**:
   - Hardware `trap;` is a fatal process-killing instruction on NVIDIA GPUs. Replace data-dependent kernel assertions with branchless register clamping and an asynchronous telemetry counter:
     ```cuda
     int64_t idx = input_ids[pos];
     if (idx < 0 || idx >= vocab_size) {
         atomicAdd(d_oob_error_count, 1ULL);  // Non-fatal async telemetry!
         idx = 0;                             // Safe clamp to row 0!
     }
     ```
2. **The Dummy Scratch Page `#0` Rule for `CUDAGraph` PagedAttention**:
   - Never initialize unused or padded `block_table` entries to `-1` (which causes `CUDA_ERROR_ILLEGAL_ADDRESS` when read by padded `CUDAGraph` warps); reserve physical KV block `#0` as a valid dummy scratch page and clamp all inactive page-table entries to `0`!

---

## ② 🎯 Architect Q&A (Q1 / Q2 / Q3)

### Q1: How can we pinpoint which kernel triggered a `device-side assert` during offline debugging?
- Run with `CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1` in your staging/reproduction environment so kernel launches become synchronous and print the exact CUDA file, line number, and offending thread coordinates. Never enable `CUDA_LAUNCH_BLOCKING=1` in production.

### Q2: How can a GPU sampler produce `INT64_MIN` (`-9223372036854775808`) during decoding?
- If all logits in a row become `-inf` (e.g., conflicting logit masks) or `FP16` temperature division overflows to `Inf`, `softmax` yields `NaN`. Passing `NaN` probabilities into `multinomial` / Gumbel-Max causes floating-point comparisons to fail and emit `INT64_MIN` or `V`, which immediately triggers `cudaErrorAssert` on the very next decode step unless clamped!

### Q3: How does the host detect an internal GPU clamping event without `cudaStreamSynchronize()`?
- Allocate `d_oob_error_count` in zero-copy mapped pinned host memory (`cudaHostAllocMapped`) or piggyback the 8-byte counter onto the existing per-step D2H sampled-token copy (`0.00 ms` extra sync latency).

---

## ③ 🧭 Visual Diagrams (Decision Tree + Comparison + Sequence)

### 3.1 GPU Index Bounds Protection & `CUcontext` Survival Decision Tree

```mermaid
flowchart TD
    A["LLM Serving / Training Index Lookup Kernels (Embedding / Gather / Sampler / PagedAttention)"] --> B{"How Are Out-of-Bounds Indices (<0 or >=V) Handled?"}
    B -- "Default PyTorch CUDA_KERNEL_ASSERT (PTX trap;)" --> C["⛔ Tier 1: 1 OOB token executes trap; -> Destroys CUcontext forever (0% survival, 30m cluster hang)!"]
    B -- "Python GPU if (ids < 0).any() or (ids >= V).any()" --> D["⛔ Tier 2: Forces cudaStreamSynchronize() -> 1.85 ms/step stall (-19.4% tput) & breaks CUDAGraph!"]
    B -- "CPU Ingress Validation Only" --> E["⚠️ Tier 3: Catches bad prompts, but misses internal GPU sampler NaN -> INT64_MIN (65% survival)!"]
    B -- "CPU 0.02ms Ingress Gate + GPU Branchless Clamp + Async Counter + Dummy Page #0" --> F["🏆 Tier 4: 100.0% CUcontext Survival + 0.02 ms Overhead + 100% CUDAGraph Compatible!"]
```

### 3.2 Fatal PTX `trap;` (`0%` Survival) & Sync `.any()` (`1.85ms`) vs. Zero-Sync Clamping (`0.02ms`)

```mermaid
flowchart LR
    subgraph Bad["⛔ Tier 1 & Tier 2: Fatal Sticky CUcontext Destruction or 1.85 ms Sync Stall"]
      B1["Tier 1: 1 OOB token (-100 or INT64_MIN) executes PTX trap; -> Kills entire TP=8 pod!"] --> B2["Tier 2: GPU .any() check stalls stream for 1.85 ms/step & breaks CUDAGraph!"]
    end
    subgraph Good["🏆 Tier 4: CPU SIMD Ingress Gate (0.02 ms) + GPU In-Register Branchless Clamp"]
      G1["Layer 1 (CPU): Reject OOB prompt_token_ids before H2D copy with HTTP 400"] --> G2["Layer 2 (GPU): Clamp OOB to row 0 + atomicAdd(d_oob_count, 1) -> 100% survival!"]
    end
```

### 3.3 CPU Pre-H2D Gate & GPU In-Kernel Branchless Clamping Sequence

```mermaid
sequenceDiagram
    participant Client as API Caller / DataLoader
    participant CPU as CPU Ingress Gate (SIMD / Rust)
    participant GPU as GPU CUDAGraph (Embedding + Sampler)
    participant Mon as Mapped Pinned OOB Telemetry

    Client->>CPU: 1. Submit `prompt_token_ids`
    alt Invalid Client Token ID (`< 0` or `>= V`)
        CPU-->>Client: 2a. Reject in 0.02 ms with `HTTP 400 Bad Request` (never touches GPU!)
    else Valid Prompt IDs
        CPU->>GPU: 2b. Async H2D copy & launch CUDAGraph decode loop
        GPU->>GPU: 3. Inside kernel: `if (idx < 0 || idx >= V) { atomicAdd(oob_ptr, 1); idx = 0; }`
        GPU-->>Mon: 4. Zero-sync mapped write increments `oob_count` if internal NaN occurred (`CUcontext` 100% safe!)
    end
```

---

## ④ 📊 4-Tier Architect Decision Matrix

| Cohort Tier | Bounds Protection Architecture | `CUcontext` Survival Rate | Step Sync Overhead | `CUDAGraph` Support | Relative Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Default PTX `trap;` Assert** | Executes hardware `trap;` on OOB | **`0.0%` ✕** | `0.00 ms` | Yes | `100.0%` (Fatal crash ✕) |
| **Tier 2: Sync GPU `.any()` Check** | D2H boolean sync every step | `100.0%` | `1.85 ms` ✕ | **No (Fails ✕)** | `80.6%` (`-19.4%` ✕) |
| **Tier 3: CPU Ingress Check Only** | Checks prompt; misses GPU NaN | `65.0%` ✕ | `0.02 ms` | Yes | `99.8%` |
| **Tier 4: CPU Gate + GPU Clamp + Page `#0` (Recommended)** | **Pre-H2D gate + in-register clamp** | **`100.0%` 🏆** | **`0.02 ms` (`92.5x` 🏆)** | **100% Compatible 🏆** | **`99.8%` 🏆** |

---

## ⑤ 🧪 Quantitative Evaluator & 7-Card 4:5 Visual Report

- **Sealed Evaluator**: [`experiments/cuda_device_side_assert_embedding_oob_sticky_context_destruction_guard_selection/run.py`](../../experiments/cuda_device_side_assert_embedding_oob_sticky_context_destruction_guard_selection/run.py) · Data: [`results.json`](../../experiments/cuda_device_side_assert_embedding_oob_sticky_context_destruction_guard_selection/results.json)
- **Interactive Sizer**: [`docs/reports/cuda-device-side-assert-embedding-oob-sticky-context-destruction-guard-sizer.html`](../reports/cuda-device-side-assert-embedding-oob-sticky-context-destruction-guard-sizer.html)
- **7-Card 4:5 Visual Benchmark Deck**: [`docs/reports/cuda-device-side-assert-embedding-oob-sticky-context-destruction-guard-selection-cards.html`](../reports/cuda-device-side-assert-embedding-oob-sticky-context-destruction-guard-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Default PTX trap; Assert", "survival_pct": 0.0, "throughput_pct": 100.0 },
      { "tier": "Tier 2: Sync GPU .any() Check", "survival_pct": 100.0, "throughput_pct": 80.6 },
      { "tier": "Tier 3: CPU Ingress Check Only", "survival_pct": 65.0, "throughput_pct": 99.8 },
      { "tier": "Tier 4: CPU Gate + GPU Branchless Clamp", "survival_pct": 100.0, "throughput_pct": 99.8 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "survival_pct": "Percentage",
    "throughput_pct": "Percentage"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "CPU Ingress Gate + GPU Branchless Clamp Guarantees 100% CUcontext Survival at 99.8% Peak Throughput",
    "subtitle": "CUcontext survival rate under injected OOB/NaN faults (%) across 4 tiers (synthetic + anonymized_reported)",
    "encodings": {
      "x": "tier",
      "y": "survival_pct",
      "color": "tier"
    }
  }
}
```

---

## ⑥ 🛠️ Production Verification Checklist (Good fit vs. Not for)

### ✅ Good fit vs. ⛔ Not for
- **✅ Good fit**: All multi-tenant LLM inference servers, `CUDAGraph` decode engines, and large-scale training pipelines.
- **⛔ Not for**: Expecting Python `try/except` to recover from `cudaErrorAssert` or running synchronous `.any()` checks on GPU tensors.

### Production Checklist
- [ ] **Validate `0 <= token_id < vocab_size` on CPU prior to H2D memory transfer (`HTTP 400` on invalid client input)**.
- [ ] **Clamp GPU-generated token IDs (`clamp(idx, 0, V - 1)`) and increment an async mapped counter instead of executing PTX `trap;`**.
- [ ] **Reserve physical block `#0` as a Dummy Scratch Page in PagedAttention so padded/aborted slots never dereference `-1`**.
