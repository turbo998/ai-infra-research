# TransformerEngine JAX Triton mHC & NCCL EP Handle Import, Megatron-LM VMM Build, LMCache MP Hooks, and Dynamo/NIXL/Mooncake Transport

> **First-Hand Upstream Sources** (`2026-10-14 ~ 2026-10-21`):
> - `NVIDIA/TransformerEngine@53d1a3c71a8f54cbfcdbc5e9f548eba040e42523` (`#3680`): *[JAX][Triton] Integrate mHC kernels (#3680)*
> - `NVIDIA/TransformerEngine@6ca7b5aa88ef2e1497f7a896ca1d9bd80442222e` (`#3639`): *[JAX] Reconstruct ncclEp Handle via ncclEpImportHandle (#3639)*
> - `NVIDIA/Megatron-LM@6c8f14a3ecbd8ac4095c5d795f2661732381d702` (`#7494`): *fix: resolve concurrent build race in vmm_symm_allocator (#7494)*
> - `LMCache/LMCache@151c6c4f415ca3040c3370c673bb2b3a0ba4646b` (`#5805`): *feat(mp): add plugin transfer extension hooks (#5805)*
> - `LMCache/LMCache@4159596c25562ba5af0c4df66d629cc253aceff1` (`#5946`): *refactor: move health_monitor under lmcache/inprocess/v1/ (#5946)*
> - `ai-dynamo/dynamo@4d31cd2f18804311bd129e9b8dc580b3d4136388` (`#16742`): *fix: release GIL during FPM relay and publisher construction (#16742)*
> - `ai-dynamo/dynamo@7929f5370eef7f15095b16f0dd41012b920c334c` (`#13744`): *feat(operator): add lossless DGDR v1beta1 <-> v1beta2 conversion (#13744)*
> - `ai-dynamo/nixl@03d9b74648cc06daef8aa71f15a53c5a98be1319` (`#2479`): *feat(tracing): add transfer request and execution tracing (#2479)*
> - `kvcache-ai/Mooncake@4137ac00260da1690bed7151af98bd2cc63f4b7d` (`#4756`): *[TE] Back Ascend host segments with mmap so A5 RoCE can drive them (#4756)*

---

## 1. TransformerEngine JAX Triton Manifold Hyper-Connections (`mHC`, `#3680`) & Stateless NCCL EP Handle Reconstruction (`#3639`)

### 1.1 Five-Stage Fused JAX Triton `mHC` Pipeline & Determinism Guard (`#3680`)

Manifold Hyper-Connections (`mHC`) maintain $n=4$ residual streams (`N_STREAMS = 4`), projecting flattened stream activations into $2n + n^2 = 24$ mixing coefficients (`N_MIX = 24`, zero-padded to `H_PADDED_DIM = 32` for power-of-two Triton tile alignment) and projecting the $4\times 4$ residual mixing block onto the Birkhoff polytope. Commit `53d1a3c71a` (`#3680`, `+3,115 / -12` lines) integrates 5 fused Triton kernel stages into JAX/Flax:
1. **Five-Stage Forward/Backward Pipeline**:
   - `mhc_projection`: Computes $\tilde{H} = X \Phi$ and per-row $\|X\|_2^2$. When `has_norm_weight=True` or (`use_tf32=False` and both $X$ and $\Phi$ are BF16), promotes $X \odot w$ and $\Phi$ to FP32 and selects **`"tf32x3"`** `tl.dot` precision (3-pass TF32 outer product emulating IEEE FP32).
   - `mhc_scale`: Fuses inverse-RMS scaling, dynamic bias addition, and `2 * sigmoid(...)` post-weights.
   - `mhc_sinkhorn`: Executes **20 log-space Sinkhorn-Knopp iterations** (`f = -logsumexp(H + g, axis=1)`, `g = -logsumexp(H + f, axis=0)`), driving bistochastic row/column sum error down to **`1.11e-7`**, and inserts `tl.debug_barrier()` before backward history readback to prevent shared-memory RAW hazards.
   - `mhc_aggregate` & `mhc_expand_combine`: Fuse multi-stream input aggregation with RMSNorm and apply residual stream mixing `x_next = H_res @ x + h_post * y`.
2. **Shardy Partitioning & Determinism**: Registers `custom_partitioning` and `SdyShardingRule` along the leading token dimension, and blocks Split-K/Split-M atomic reductions when `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`.

### 1.2 Per-Step `ncclEpImportHandle` Reconstruction in JAX (`#3639`)

Across `jax.lax.scan`-stacked MoE layers, XLA relocates and reuses temporary buffers, causing pointer-keyed `EpHandleCache` lookups to alias stale `ncclEpHandle_t` handles from earlier layers. Commit `6ca7b5aa88` (`#3639`) introduces `nvte_ep_{dispatch,combine,dispatch_bwd,combine_bwd}_v2`, which reconstructs an owned `ScopedHandle(handle, /*owned=*/true)` on every step from `handle_mem` bytes + `EpLayerConfig` via `ncclEpImportHandle` and destroys it upon completion, while validating `topk_idx_dtype` (`INT32` or `INT64`) in C++ and retaining a bounded LRU cache (`NVTE_EP_HANDLE_CACHE_SIZE`) for PyTorch.

---

## 2. Megatron-LM Concurrent VMM Allocator Build Fix (`#7494`) & LMCache MP Plugin Hooks (`#5805`, `#5946`)

1. **Megatron-LM VMM Allocator Build Race & Fabric Fallback (`#7494`)**:
   - Replaces `torch.utils.cpp_extension.load_inline` (which overwrote `main.cpp` in the shared build directory before acquiring PyTorch's build file lock) with `cpp_extension.load` over packaged `megatron/core/allocator/csrc/vmm_symm_allocator.cpp`, caching both `_allocator` and `_build_error`.
   - When `CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED=1` is reported on nodes without a provisioned IMEX daemon, automatically strips `CU_MEM_HANDLE_TYPE_FABRIC` (`0x8`) after the initial `cuMemCreate` failure and retries with `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` (`0x1`).
2. **LMCache Multiprocess (`mp`) Plugin Transfer Extension Hooks (`#5805`, `#5946`)**:
   - Adds `create_lmcache_mp_worker_adapter`, `_create_transfer_context`, `_create_cache_context`, `BaseCacheContext.transfer_object_group`, and `CUDA_CacheContext._make_temp_buffer(..., full_sw_kv=full_sw_kv)`; pads skipped prefix chunks with `[None] * skip` during `retrieve` with `skip_first_n_tokens > 0`; and moves `HealthMonitor` check discovery under `lmcache.inprocess.v1.health_monitor.checks`.

---

## 3. Dynamo FPM GIL Release & DGDR `v1beta2` Conversion (`#16742`, `#13744`), NIXL Request Tracing (`#2479`) & Mooncake Ascend A5 `memfd` (`#4756`)

- **Dynamo (`#16742`, `#13744`)**: Wraps `rt.block_on(...)` inside `py.allow_threads` during `FpmEventRelay::new` and `FpmDirectPublisher::new` to eliminate GIL-vs-Tokio callback deadlocks; and embeds mutually exclusive `v1beta2` conversion envelopes inside `v1beta1` hub and `v1alpha1` spoke CRDs for lossless carrier-free round-tripping.
- **NIXL (`#2479`)**: Attaches a non-zero 64-bit unsigned decimal `nixl.xfer_request_id` (redrawing on random `0`) and a monotonic `nixl.xfer_execution` counter (incremented only after repost/notification validation succeeds) to transfer spans and `nixl::xfer.complete` events.
- **Mooncake (`#4756`)**: Backs Ascend A5 `roce:host` segments with `2 MiB` THP-aligned `mmap` / shareable `memfd` (`host_memfd.h`), rebases cross-process addresses via `SharedHostMapper::Translate` (`local_base + (dest_addr - owner_base)`), and treats `ACL_MEM_LOCATION_TYPE_NONE` (`UNREGISTERED`) imported `memfd` buffers as host memory so `LocalCopyEngine` dispatches `aclrtMemcpyBatch`.
