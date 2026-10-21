# Megatron-LM GDN Single-Update CUDA Graph Replay, TorchTitan Optimizer-Owned GradientClipper & LoRA TP SPMD Typecheck, LMCache Dynamic L2 Prefetch Watermark & VMM Padded-Pool IPC, and Mooncake / NIXL / Dynamo Transport Hardening

> **First-Hand Upstream Sources**:
> - `NVIDIA/Megatron-LM@29e1a7b6697c61d184c95726865d417fc51b8574` (`fix(inference): model GDN prefill CUDA graph as single-update GDP (#8353)`, 2026-10-21)
> - `pytorch/torchtitan@9698a733623ca997ef824a03cb1f44706ee631e7` (`Move gradient clipping into OptimizersContainer as a GradientClipper component (#5394)`, 2026-10-21)
> - `pytorch/torchtitan@a242475253433725f10f39c18f5ccbfdf188e097` (`Fix LoRA compatibility with SPMD type checking under TP (#5390)`, 2026-10-21)
> - `LMCache/LMCache@a6accee6d84c77da8c8b3e3a68574b69247e9684` (`perf(mp): replace fixed max-in-flight with dynamic staging memory watermark for L2 prefetch (#5889)`, 2026-10-21)
> - `LMCache/LMCache@21aa6ac06c83e1bad460cc231c5168a6bc7defe6` (`fix: register dim-0-padded KV pools in VmmCudaIPCWrapper (#5855)`, 2026-10-21)
> - `LMCache/LMCache@45486917f13aa454a22435aa03e2dc28efda47f5` (`fix(c_ops): ensure primary CUDA context on current thread in stream_ops (#5929)`, 2026-10-21)
> - `kvcache-ai/Mooncake@7a0585567458a4aba6a58115888c54b7fd1a44e1` (`Allow CPU clients to select peer GPU TCP memory (#4410)`, 2026-10-20)
> - `ai-dynamo/nixl@03d9b74648cc06daef8aa71f15a53c5a98be1319` (`feat(telemetry): add xfer request ID and execution counter to transfer requests (#2479)`, 2026-10-21)
> - `ai-dynamo/dynamo@5004f3fa8093d99de68a8054ea238ea534d9daf4` (`fix(multimodal): propagate errors in count_all_image_tokens (#16947)`, 2026-10-20)

---

## 1. Megatron-LM GDN Single-Update CUDA Graph Replay & TorchTitan Training Contracts (`Megatron-LM@29e1a7b669`, `torchtitan@9698a73362`, `torchtitan@a242475253`)

### 1.1 Modeling Gated DeltaNet (GDN) Prefill CUDA Graph as Single-Update GDP (`Megatron-LM@29e1a7b669`)
In Megatron-LM's hybrid inference context, GDN prefill runs `chunk_gated_delta_rule` with fixed `GDP_CHUNK_SIZE = 64`. For CUDA Graph capture and replay, the kernel must read chunk boundaries from a persistent static device tensor `cu_seqlens`:
- Because GDN prefill is mathematically equivalent to a Generalized Delta Product (GDP) with `num_householder = 1` reflection vector per 64-token chunk, commit `29e1a7b669` models GDN prefill CUDA Graph buffers with `num_householder = 1`, preallocating `4096 // 64 = 64` persistent chunk descriptors (`cu_seqlens` length `65` at `max_tokens = 4096`) and eliminating dynamic buffer shape mismatches during graph replay.

### 1.2 TorchTitan Optimizer-Owned `GradientClipper` & LoRA TP SPMD Typecheck (`torchtitan@9698a73362`, `torchtitan@a242475253`)
- **Optimizer-Owned `GradientClipper` (`torchtitan@9698a73362`)**: Moves gradient clipping into `OptimizersContainer` as a first-class `GradientClipper` component invoked inside the optimizer `_update` hook during `step()`. This decouples `Trainer.forward_backward_step` from manual clipping and ensures `FTManager` (TorchFT fault-tolerant training) automatically preserves full multi-dimensional (`DP/PP/EP`) total-norm reduction semantics when overriding the optimizer step hook.
- **LoRA Compatibility with TP SPMD Typechecking (`torchtitan@a242475253`)**: Under Tensor Parallelism, `RowwiseParallel` on `lora_a` produces a `Partial` activation tensor that is fed into `lora_b` (whose weight is `Replicate` when the base layer's output stays `Partial`). Previously, `_LinearFunction.spmd_typecheck` rejected `Partial` inputs; delegating `_LinearFunction.spmd_typecheck` to the `einsum("...k,nk->...n")` bilinearity rule validly propagates `Partial @ Replicate -> Partial` before the deferred all-reduce at the base layer output addition.

---

## 2. LMCache Dynamic L2 Prefetch Watermark, VMM Padded-Pool IPC & Driver Context Binding (`LMCache@a6accee6d8`, `LMCache@21aa6ac06c`, `LMCache@45486917f1`)

### 2.1 Dynamic `--l2-prefetch-inflight-watermark` Admission (`LMCache@a6accee6d8`)
Previously, `PrefetchController` gated concurrent `L2_LOADING` tasks with a fixed count `max_in_flight = 8`:
- Small `8 MiB` prefetch tasks underutilized a `1024 MiB` pinned-memory staging pool (`64 MiB = 6.25%`), while raising the fixed count risked exhausting the staging pool on large `128 MiB` prefetches and starving foreground `store` operations;
- Commit `a6accee6d8` replaces the fixed counter with `--l2-prefetch-inflight-watermark` (default `0.5`, i.e., `512 MiB`): tasks are admitted while `inflight_bytes + req_bytes <= pool_capacity * watermark` (plus a deadlock-free single-request bypass when `inflight_bytes == 0` and `req_bytes <= pool_capacity`), delivering **4.0x higher concurrency (`32 vs 8`)** for `8 MiB` requests while capping `128 MiB` requests at `4` (`512 MiB`).

### 2.2 `VmmCudaIPCWrapper` Dim-0 Padded Pool Span & `stream_ops` Primary Context Binding
- **Dim-0 Padded KV Pool IPC (`LMCache@21aa6ac06c`)**: vLLM's cross-layer uniform KV cache pads dimension 0 so `stride(0) > prod(shape[1:])` (e.g., shape `(2, 8, 64)` with `stride(0) = 1280` instead of `512`). `VmmCudaIPCWrapper` previously walked VMM physical allocations using `numel() * element_size() = 2048 B`, missing the trailing strided span; computing `1 + sum((shape[i] - 1) * stride[i]) = 1792 elements = 3584 B` covers the entire padded allocation.
- **Primary Context Binding in `stream_ops` (`LMCache@45486917f1`)**: CUDA Driver APIs `cuStreamWaitValue64` and `cuStreamWriteValue64` require an active `CUcontext` on the calling OS thread (`CUDA_ERROR_INVALID_CONTEXT = 201` on fresh worker threads). `ensure_cuda_ctx()` queries `cudaGetDevice(&dev)`, retains the primary context (`cuDevicePrimaryCtxRetain`), and binds it (`cuCtxSetCurrent`) when `cuCtxGetCurrent` is null.

---

## 3. Mooncake Hybrid TCP Transport, NIXL Request Telemetry & Dynamo Multimodal Error Propagation

| Component & Commit | Gap / Failure Mode | Architectural Fix & Quantitative Impact |
| :--- | :--- | :--- |
| **Mooncake CPU↔GPU TCP Selection** (`Mooncake@7a05855674`) | CPU-only clients lacked local CUDA devices, leaving `caps.dram_to_gpu` and `caps.gpu_to_dram` false and rejecting remote peer `cuda:*` TCP memory segments. | Enables `caps.dram_to_gpu = true` and `caps.gpu_to_dram = true` whenever TCP transport is installed (since the remote GPU peer performs the local `cudaMemcpy`), allowing CPU clients to read/write peer GPU TCP KV buffers. |
| **NIXL 64-Bit Request ID & Execution Counter** (`nixl@03d9b74648`) | Re-posting a `nixlXferReqH` handle via `postXferReq` produced ambiguous telemetry events across retries/re-executions. | Assigns a process-unique atomic 64-bit unsigned decimal `nixl.xfer_request_id` at handle creation and increments `nixl.xfer_execution` (`1 -> 2 -> 3`) on every `postXferReq`. |
| **Dynamo Multimodal Token Error Propagation** (`dynamo@5004f3fa80`) | `count_all_image_tokens` swallowed decoder/shape errors and returned `0`, setting `image_tokens = Some(0)` and under-accounting KV routing load. | Changes `count_all_image_tokens` to return `Result<usize>` and logs a `warn!` while setting `image_tokens = None` on failure, preventing bogus zero-token routing decisions. |

---

## 4. Deterministic Reproduction (`experiments/megatron-gdn-cudagraph-titan-gradclip-lmcache-prefetch-vmm-mooncake-nixl/run.py`)

```bash
python experiments/megatron-gdn-cudagraph-titan-gradclip-lmcache-prefetch-vmm-mooncake-nixl/run.py --check
```
