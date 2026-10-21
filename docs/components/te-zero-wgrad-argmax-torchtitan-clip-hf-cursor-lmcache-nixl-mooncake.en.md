# TransformerEngine Zero-Storage Dummy `wgrad` & NaN-Safe TP Argmax, TorchTitan `GradientClipper` & HF Cursor Preservation, LMCache Affinity RPC, NIXL UCX CPU Proxy & Mooncake Store Safety

> **Phase 52 (`priority: 61`) · Track 3 First-Hand Source Analysis**  
> **Upstream Commits**: `NVIDIA/TransformerEngine@fc48751409dd` (`#3670`), `39a8b879dfbf` (`#3671`), `e6afa9ddfa81` (`#3606`); `pytorch/torchtitan@9698a733623c` (`#5394`), `c0eb13609240` (`#5501`); `LMCache/LMCache@b28250e765c8` (`#5803`), `fcee51cc5e01` (`#5903`); `ai-dynamo/dynamo@4f42be7bebd9` (`#16974`); `ai-dynamo/nixl@3f6e1a2716e7` (`#2327`); `kvcache-ai/Mooncake@631e1019cf88` (`#4622`), `1a6d00d0f4dc` (`#4688`), `3cf058db6d42` (`#4673`), `d782a5998f6d` (`#4755`)  
> **Deterministic Reproduction**: `experiments/te-zero-wgrad-argmax-torchtitan-clip-hf-cursor-lmcache-nixl-mooncake/run.py --check`

---

## 1. TransformerEngine Zero-Storage Dummy `wgrad` (`#3670`), NaN-Safe TP Cross-Entropy Argmax (`#3671`), and `GroupedLinear` Refactor (`#3606`)

1. **`zero_tensor_dummy_wgrad` & Per-Device Cache (`#3670`)**:
   - Adds `zero_tensor_dummy_wgrad: bool = False` to `get_dummy_wgrad` (enabled in `BasicLinear._get_dummy_wgrad` when `zero_out_is_needed` is false), allocating zero physical storage (`0 bytes`) via `torch._efficientzerotensor(shape, dtype=dtype, device=device)` and caching by `(*shape, dtype, str(device))`. Across 32 layers of `4096 × 14336` BF16 weights, this eliminates **`3.76 GB`** of dummy buffer allocations.
2. **NaN-Safe Tie-Breaking `parallel_cross_entropy(..., return_argmax=True)` (`#3671`)**:
   - Because Triton's `tl.max(..., return_indices=True)` evaluates comparisons with `NaN` as false, `_fold_block_argmax` tracks the first `NaN` separately from non-NaN maxima and packs each TP rank's local index as raw `FP32` bits into an extra column of the existing statistics all-gather, matching `torch.argmax` over unsharded logits (first `NaN` wins, else first global maximum wins) with bit-identical loss and gradients.
3. **`GroupedLinear` Shared Backward State Holder (`#3606`)**: Extracts `y.grad_fn.backward_objects` across split-quantize and grouped-tensor paths and captures only the accumulator flag in deferred fused `wgrad` callbacks so cast weights are freed immediately after `dgrad`.

---

## 2. TorchTitan Optimizer-Owned `GradientClipper` (`#5394`) & HF Streaming Cursor Preservation (`#5501`)

1. **Configurable `GradientClipper` (`#5394`)**: Moves gradient clipping into an optimizer-owned `GradientClipper(Configurable)` shared by `Optim` and `TorchFTOptim`.
2. **`_HuggingFaceCursorIterator._pending_hf_state` (`#5501`)**:
   - Hugging Face's `IterableDataset.load_state_dict()` applies a restored cursor lazily on the next read (`next()`). When a document-aware packer snapshots its source after `set_state()` but before reading a row, `get_state()` previously returned the stale pre-restore position (`0` instead of `3`), corrupting a second resume (`[14..17]` instead of `[24..27]`).
   - Caching the restored bytes in `self._pending_hf_state` until `__next__` yields a row preserves exact state across back-to-back checkpoints and failed reads.

---

## 3. LMCache Affinity RPC (`#5803`, `#5903`), Dynamo SGLang 2s Pool Timeout (`#16974`), NIXL UCX CPU Proxy (`#2327`), and Mooncake Store Invariants (`#4622`, `#4688`, `#4673`, `#4755`)

1. **LMCache `server_module_affinity_call` (`#5803`) & Pickle Ban (`#5903`)**: Orders `requires_client_affinity=True` plugin handlers on the client's affinity thread alongside `store`/`retrieve`, and enforces Ruff `TID251` (`descriptor_payload` MessagePack allowlist).
2. **Dynamo SGLang `pool_idle_timeout(2s)` (`#16974`)**: Retires idle HTTP connections in `reqwest::Client` after 2s, before SGLang's 5s keep-alive closes them.
3. **NIXL `nixl::ucx::proxyEngine` (`device_proxy=true`, `#2327`)**: Partitions shared UCX workers one per `(channel, peer)` ring (`128` workers for `32 × 4`), fences each worker before `ucp_atomic_op_nbx` (`UCP_ATOMIC_OP_ADD`), and enforces `NIXL_THREAD_SYNC_RW` with `ucx_error_handling_mode=none`.
4. **Mooncake Store & Reshard Invariants (`#4622`, `#4688`, `#4673`, `#4755`)**: Rejects foreign Store registration leases (`_owner is not self`), snapshots `statx` `STATX_BTIME` into `TimeKey` before `std::sort`, adds `MOONCAKE_OFFLOAD_FLUSH_PARTIAL_BUCKETS=1` same-cycle partial bucket flush, and validates cached DFS shard descriptors (`shard->fd >= 0 && shard->path == descriptor.file_path`).
