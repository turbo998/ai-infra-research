# FlashInfer cuDNN Prefill Descriptor Reuse, Split-K Async-Proxy Fence & TileLang NVRTC/Vectorize Analyzer Isolation

> **Phase 66 · Priority 75 · Track A (GPU Kernels, Compilers & Quantization Acceleration)**  
> **Upstream Commits Covered**: `flashinfer-ai/flashinfer@b27f6cae7da4`, `flashinfer-ai/flashinfer@c56cd4fb7222`, `flashinfer-ai/flashinfer@c01aa7779bbb`, `flashinfer-ai/flashinfer@6b38488084e3`, `tile-ai/tilelang@35ee29507557`, `tile-ai/tilelang@fe8c34618ffb`  
> **Companion Deterministic Experiment**: `experiments/flashinfer-cudnn-prefill-splitk-fence-tilelang-nvrtc/run.py` / `summary.json`

---

## 1. Architectural Context & Core Engineering Problems

In high-throughput long-context serving and Mixture-of-Experts (MoE) training across Blackwell (`sm_100a`/`sm_103a`/`sm_107a`) and Hopper (`H100 NVL`), host launch overhead, asynchronous shared-memory proxy visibility, and JIT cache invariants directly govern latency and numerical correctness:

1. **cuDNN Prefill Plan Descriptor Rebuild Overhead & Retained Metadata Mutation Hazard**: In `flashinfer/cudnn/prefill.py`, unconditionally computing `_prefill_descriptor_key(**metadata.graph_kwargs(None))` on every `_CudnnPrefillPlan.prepare()` call wastes host CPU cycles inspecting unchanged tensor shapes and strides. Worse, the legacy constructor mutated the caller's `metadata` object in place via `setattr(metadata, name, views[id(tensor)])`, causing subsequent in-place layout changes (such as `metadata.block_tables.as_strided_()`) on a reused metadata instance to alias the plan's snapshot and go undetected.
2. **Blackwell Dense-Projection GEMM Split-K Collector (`_sks_sb2`) Async-Proxy Ordering Hazard**: In `flashinfer/experimental/dense_projection_gemm`, the 6 split-K 2-slab collector kernels read partial sums from `shared::cta` via the generic proxy after asynchronous proxy writes complete. Without an explicit `fence.proxy.async.shared::cta;`, generic loads can race ahead of async-proxy stores on Blackwell SMs.
3. **CAKE MoE Grouped GEMM FP32 WGrad Source Duplication**: Maintaining separate generated CUDA source files for BF16 and FP32 weight-gradient (`wgrad`) outputs duplicated 3 entire kernel programs (`3,125` net lines).
4. **TileLang NVRTC Disk-Cache `libpath` Omission & `VectorizeLoop` Analyzer Pollution**: `NVRTCAdapter.from_database` forgot to copy `adapter.lib_generator.libpath` onto `adapter.libpath`, breaking `@autotune` result saving after disk-cache hits; meanwhile, `VectorizeLoop` in `src/transform/loop_vectorize.cc` passed the caller's `arith::Analyzer*` directly into `VectorizeRewriter`, leaking loop-local arithmetic bindings into subsequent transformations.

---

## 2. First-Hand Implementation Paths & Fixes

### 2.1 FlashInfer `_CudnnPrefillPlan.prepare()` Scalar Fast Path & Private Owned Snapshots (`flashinfer@b27f6cae7da4`)

In `flashinfer/cudnn/prefill.py`, `_CudnnPrefillPlan.prepare(cls, metadata, dtype, device, previous=None)` implements a two-tier reuse check:

- **Tier 1 (Scalar & Graph Invariant Check)**: Validates `previous.device == device`, `previous.dtype == dtype`, `previous.metadata.o_data_type == metadata.o_data_type`, `previous.override_enabled == enabled`, `previous.metadata._bounded_mla == metadata._bounded_mla`, and checks `(metadata.max_token_per_sequence, metadata.max_sequence_kv)`, `causal`, and `return_lse` directly against `previous.exact_keys[True]`.
- **Tier 2 (Private Snapshot Storage & Layout Verification)**: Iterates over `zip(previous.inputs, inputs, strict=True)`:
  - When `old is new` or `old.is_set_to(new)` (identical underlying storage and offset), `same_bindings` remains `True` and `previous` is returned immediately.
  - When storage pointers change but `old.shape == new.shape` and `old.stride() == new.stride()`, `same_bindings` becomes `False` while `same_descriptor` remains `True`, reusing `previous.exact_keys[True]` without calling `_prefill_descriptor_key(...)`.
  - In `__init__`, detached tensor views are stored privately in `self.inputs` and bound to a cloned `metadata = replace(metadata, ...)` rather than mutating the caller's `metadata` in place. Consequently, an in-place `metadata.block_tables.as_strided_((2, 4), (1, 2))` on a retained metadata object is deterministically caught on the next `prepare()`.

**Author-Benchmark Results (H100 NVL, `BF16 D128, Hq/Hkv=32/8`)**:
- `B=1, Q=128, KV=8192`: Host `plan()` latency drops from `79.12 us` to `68.12 us` (**`-13.90%`**), and `plan+run` drops from `137.36 us` to `126.12 us` (**`-8.18%`**).
- `B=4, Q=512, KV=512`: Host `plan()` latency drops from `77.38 us` to `65.83 us` (**`-14.93%`**), and `plan+run` drops from `135.27 us` to `122.01 us` (**`-9.80%`**).

### 2.2 Split-K Slab Collector Async-Proxy Fence & 247-Entry Refresh (`flashinfer@c56cd4fb7222`)

In `flashinfer/experimental/dense_projection_gemm`:
- All 6 `_sks_sb2` split-K slab collectors (including `cake_dense_projection_gemm_2307cb3cca86763a9027`, template `dense_proj_gemm_nn_n256_sks_sb2_pro7` with `sk_slab=2, sk_sync=True`) now emit `fence.proxy.async.shared::cta;` before bulk slab reads.
- 13 BF16-output and 3 FP32-output kernels refresh their `st.global.L1::no_allocate.L2::evict_first.v8.b32` fused split store lowerings across all `247` build entries (`80` on `sm_100a`, `80` on `sm_103a`, `87` on `sm_107a`), delivering **`1.008x–1.016x`** speedups on GB300 and **`1.003x–1.037x`** on GB200 across `dense_gate_up`, `dense_down`, and `5120 -> 5120` BF16 wgrad benchmarks.

### 2.3 CAKE MoE `WGRAD_OUT=32` Specialization & Unreachable Softmax Pruning (`flashinfer@c01aa7779bbb`, `flashinfer@6b38488084e3`)

- `flashinfer@c01aa7779bbb` folds 3 FP32 wgrad MoE grouped GEMM clones onto their BF16 counterparts using compile-line specialization (`-DWGRAD_OUT=16` vs `-DWGRAD_OUT=32` alongside `-DWGRAD_TILE={256,512}`), shrinking `PROGRAMS` from `8` to `5`, removing **`3,125` net lines**, and preserving **bitwise-identical SASS** across all `26` compiled libraries.
- `flashinfer@6b38488084e3` removes 3 unreachable kernels in `cake_blackwell_softmax` (**`-1,829` net lines**, **`4,954` combined lines removed**).

### 2.4 TileLang NVRTC `libpath` Restoration & Cloned Analyzer in `VectorizeLoop` (`tilelang@35ee29507557`, `tilelang@fe8c34618ffb`)

- `NVRTCAdapter.from_database` in `tilelang/jit/adapter/nvrtc/adapter.py` assigns `adapter.libpath = adapter.lib_generator.libpath` after loading a cached library, fixing autotune artifact persistence on NVRTC disk-cache hits.
- `VectorizeLoop` in `src/transform/loop_vectorize.cc` clones the incoming analyzer (`auto local_analyzer = analyzer->Clone();`) before invoking `VectorizeRewriter`, preventing loop-local domain bindings from polluting sibling loops.

---

## 3. Deterministic Verification Matrix

Running `python experiments/flashinfer-cudnn-prefill-splitk-fence-tilelang-nvrtc/run.py --check` validates:

| Dimension | Legacy Baseline | Phase 66 Fix | Quantitative Impact / Architectural Benefit |
| :--- | :--- | :--- | :--- |
| **cuDNN Prefill `plan()` (B4, Q512, KV512)** | `77.38 us` | `65.83 us` | **`14.93%`** host plan reduction (**`9.80%`** `plan+run` reduction) |
| **Retained `metadata` In-Place `as_strided_`** | Hidden by in-place `setattr` | Detected via private `detach()` views | `100%` detection of layout/stride mutations |
| **Split-K Slab Collectors (`_sks_sb2`)** | Unfenced generic read after async write | `fence.proxy.async.shared::cta;` | `247` entries (`80/80/87`), up to **`1.037x`** faster on GB200 |
| **CAKE MoE WGrad + Softmax Codebase** | `8` WGrad entries + 3 dead softmax | Folded to `5` via `WGRAD_OUT={16,32}` | **`-4,954` net lines**, bitwise-identical SASS across `26` libs |
| **TileLang NVRTC & `VectorizeLoop`** | `libpath=None`; shared `arith::Analyzer` | `libpath` restored; `analyzer->Clone()` | Zero autotune cache-hit crash; `0` polluted analyzer bindings |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Calculation Exercise**: In a chunked-prefill workload executing `2,500` steps/sec at `B=4, Q=512, KV=512` with unchanged tensor shapes/strides, how much host CPU time per second is saved by `_CudnnPrefillPlan.prepare()`? (`2,500 * 11.55 us = 28.875 ms/s`, or `2.89%` of a single host core).
2. **Falsifiable Hypothesis**: If a caller rebinds a tensor to a newly allocated buffer via `set_()` with the exact same `shape` and `stride`, does `_CudnnPrefillPlan.prepare()` recompute `_prefill_descriptor_key()`?  
   - **Verification**: **No**. `same_descriptor` remains `True` (reusing `previous.exact_keys[True]`), while `same_bindings` becomes `False` so only the pointer bindings are refreshed.
