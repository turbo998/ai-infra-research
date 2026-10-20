# PyTorch CUDA Copy-on-Write (`COW`) Explicit Completion Events, CUDA Graph Capture Boundary Hardening & `c10d` `ProcessGroupNCCLLazy` MemPool Forwarding

> **中文版**: [pytorch-cuda-cow-completion-events-c10d-nccl-lazy-mempool.md](pytorch-cuda-cow-completion-events-c10d-nccl-lazy-mempool.md)  
> **Evidence Class**: `implementation` (verified against upstream commits) + `own-offline` (`experiments/pytorch_cuda_cow_nccl_lazy_mempool/run.py`)

---

## 1. Architectural Problem Statement

PyTorch's Copy-on-Write (`COW`) tensor storage (`_lazy_clone`) elides eager memory copies until a mutable write triggers `materialize_cow`. Prior to `pytorch/pytorch` PRs `#200800` and `#201432`, three architectural flaws affected CUDA execution and `nccl-lazy` symmetric memory pools:

1. **Unordered Default-Stream `cudaMemcpy` Breaks CUDA Graph Replay**: `NativeCachingAllocator::copy_data` issued a synchronous `cudaMemcpy` on the legacy default stream, which was unordered against non-blocking streams and omitted from CUDA Graph capture (`8/8` stale replay reads).
2. **2x Memory Retention & Unrelated Stream Stalls on Last-Reference Steal**: Completing a materializing copy on operation stream $B$ does not guarantee that earlier asynchronous reads on side streams have finished. Conservatively retaining the old reference in `StorageExtraMeta::cow_retired_data` doubles peak memory (`2.0x`), while stealing the last reference without completion tracking (`needs_stream_wait_ = true`) forces waiting on the entire allocation stream $A$ (`45.0 ms` including unrelated later kernels).
3. **Missing `register_mem_pool` / `deregister_mem_pool` on `ProcessGroupNCCLLazy`**: `ProcessGroupNCCLLazy` forwarded `mem_allocator` and windows to its primary communicator (`getPrimary()`), but omitted `register_mem_pool` and `deregister_mem_pool` bindings.

---

## 2. First-Hand Upstream Implementation Verification (`implementation`)

### 2.1 Allocation-Stream Materialization, `cow_retired_data` & `_cow_record_done` (`pytorch/pytorch` `20f82b619b`, `#200800`)

In `c10/core/impl/COW.cpp` (blob `70357669379b557e585316deef6d661397f0b02b`), `c10/core/impl/COWDeleter.cpp` (blob `bda590c8912a31b0daea6154c353daecc124043d`), `c10/core/impl/COWDeleter.h` (blob `1b3c27c92cdd88b8519d7a3c3bd8a54ec8d287ec`), `c10/core/impl/README-cow.md` (blob `5089906503639d62c4523f4be281c5acb7def821`), and `c10/cuda/CUDACachingAllocator.cpp` (blob `d1e76491796982f1ac278c1dbc858fba9164f2c7`):

- **Allocation-Stream Clone & Event Ordering**: `NativeCachingAllocator::copy_data` uses `cudaMemcpyAsync` on `cuda::getCurrentCUDAStream()`. When a shared COW tensor materializes on stream $B$, `materialize_cow` switches to `*ctx->stream()` (allocation stream $A$), clones the buffer on $A$, calls `impl->waitStream(stream, *ctx->stream())`, and stores the old `DataPtr` in `storage->get_extra_meta().cow_retired_data`.
- **Explicit Completion via `torch._C._cow_record_done(tensor)`**: `COWReference::record_done()` records a `COWCompletion` event on the current stream, blocks `reuse_stream` (`done->event.block(reuse_stream)`), and resets `cow_retired_data`. When the last remaining sibling (`ctx->is_unique()`) materializes and all departed siblings recorded completion (`needs_stream_wait_ == false`), it steals the backing allocation in-place after waiting only on the recorded `completions_` events.
- **CUDA Graph Capture Boundary Enforcement**: `check_cow_read`, `lazy_clone_storage`, `materialize_cow`, and `record_done` enforce `capture_id == ctx->capture_id()`, rejecting cross-capture shared materialization, cross-capture completion, and unmaterialized external COW graph inputs while falling back to `eager_clone` when cloning across streams or private pools.

### 2.2 `ProcessGroupNCCLLazy` MemPool Forwarding (`pytorch/pytorch` `47bc8be46b`, `#201432`)

In `torch/csrc/distributed/c10d/init.cpp` (blob `f2314ee6a8f0603b324cef670039d103cf06be8b`) and `test/distributed/test_c10d_nccl2.py` (blob `820b9915c9b422b2324e3a091878a1b85e8fe0c6`), `ProcessGroupNCCLLazy` binds `register_mem_pool(pool, symm=false)` and `deregister_mem_pool(pool)` by delegating to `self.getPrimary()->registerMemPool(pool, symm)` and `self.getPrimary()->deregisterMemPool(pool)`.

---

## 3. Deterministic Standard-Library Verification (`own-offline`)

Run `python experiments\pytorch_cuda_cow_nccl_lazy_mempool\run.py --check`:

| COW Execution Mode | Stale CUDA Graph Replay Reads | 2-Sibling Peak Memory (`MiB`) | Retained `cow_retired_data` | Last-Sibling Steal Stall (`ms`) | Extra D2D Copies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Legacy Default-Stream `cudaMemcpy` | `8 / 8` (100% stale) | `512.0 MiB` (`2.00x`) | N/A | `45.0 ms` | `1` |
| Conservative Event COW (no `_cow_record_done`) | **`0 / 8`** | `512.0 MiB` (`2.00x`) | `256.0 MiB` | `45.0 ms` (`needs_stream_wait_=true`) | `1` |
| **Explicit-Completion COW (`_cow_record_done` Steal)** | **`0 / 8`** | **`256.0 MiB` (`1.00x`, `-50.0%`)** | **`0.0 MiB`** | **`10.0 ms` (`-77.78%` stall reduction)** | **`0` (`4.0 ms` saved)** |
