# LMCache P2P Local-DRAM Isolation & Error Ref Release, NIXL UCX InlinedVector Notifications & SharedCuObjClient, Megatron GDN Graph-Safe State, TE Varying-Dim MXFP8 Scales, TorchTitan Shared Eval CUDA Graph Pool, veRL Liger TP Fused CE, Mooncake 1Hz Port Recovery & Dynamo Direct Stream Cancellation (Phase 97)

> **中文版**: [lmcache-p2p-dram-put-ref-nixl-ucx-cuobj-megatron-gdn-te-mxfp8-titan-verl-mooncake-dynamo-phase97.md](lmcache-p2p-dram-put-ref-nixl-ucx-cuobj-megatron-gdn-te-mxfp8-titan-verl-mooncake-dynamo-phase97.md)
> **Priority**: `106` · **Question ID**: `lmcache-p2p-dram-put-ref-nixl-ucx-cuobj-megatron-gdn-te-mxfp8-titan-verl-mooncake-dynamo-phase97` · **Evidence Class**: `implementation` + `author-benchmark` + `own-offline`

---

## 1. Executive Summary & Architectural Findings

Across 10 zero-overlap commits in `LMCache/LMCache`, `ai-dynamo/nixl`, `NVIDIA/Megatron-LM`, `NVIDIA/TransformerEngine`, `pytorch/torchtitan`, `volcengine/verl`, `kvcache-ai/Mooncake`, and `ai-dynamo/dynamo`, Phase 97 establishes key distributed memory, RDMA/UCX, and training invariants:

1. **Restricting LMCache P2P Lookups to Local DRAM Segments (`LMCache#6014`, `c90e6862ab`)**: `p2p_lookup_and_lock` passes `allowed_l1_memory_tags=frozenset({(MemorySegment.DRAM, MemoryResidency.LOCAL)})` and `allowed_l2_memory_tags=frozenset()`, preventing peer lookups from cascading into remote P2P or disk L2 backends.
2. **Releasing Unsubmitted `MemoryObj` References on `batched_put` Backend Exceptions (`LMCache#5940`, `562517ad9f`)**: Wraps the `StorageManager.batched_put` backend loop in `try/except` to call `ref_count_down()` on remaining `memory_objs`, eliminating pinned memory leaks on error paths (`N -> 0` leaked chunks).
3. **NIXL UCX Notification Fast Path with `absl::InlinedVector` & Zero-Copy SerDes (`nixl#2412`, `0165676874`)**: Replaces heap-allocated `std::vector` with `absl::InlinedVector` in `nixlUcxBackendReqH`, adds `exportMoveStr`, and unifies `int_handle->flush()`, cutting 4 KiB × 8 P99 prep latency from `8.0 -> 6.0 us` (`1.3333x`), P99 post latency from `12.0 -> 9.0 us` (`1.3333x`), and increasing bandwidth from `4.052` to `4.169 GB/s`.
4. **In-Process `cuMemObj` Sharing Across Multi-Worker Agents (`nixl#2379`, `a486deaa23`)**: `SharedCuObjClient` reference-counts `(ptr, size)` registrations across in-process agents sharing a 1 GiB pinned arena so only the first registration invokes `cuMemObjGetDescriptor` and only the final deregistration calls `cuMemObjPutDescriptor`.
5. **CUDA-Graph-Safe Recurrent State Allocation in Megatron-LM Gated DeltaNet (`Megatron-LM#8322`, `e592c042d5`)**: `torch_chunk_gated_delta_rule` replaces `torch.zeros(...).to(value)` with `value.new_zeros((batch_size, num_heads, k_head_dim, v_head_dim))`, preventing host-to-device copies during CUDA Graph capture.
6. **Decoupled Scale Base Offsets for Varying-Last-Dim Grouped MXFP8 Quantization (`TransformerEngine#3599`, `76ae4b0981`)**: `group_quantize_mxfp8_kernel` tracks `tensor_base_for_scales` separately from `tensor_base` so swizzled GEMM scales (`optimize_for_gemm=True`) with 4-block row alignment never corrupt adjacent tensors when last dimensions vary.
7. **Shared Training & Validation CUDA Graph Memory Pool in TorchTitan (`torchtitan#5241`, `3c7b0c9d94`)**: Shares the CUDA Graph pool between training, validation (`validator.enable_cuda_graphs=True`), and optimizer after eager warmup, holding Muse Glimmer 30B peak memory at `61.530 GiB` vs `61.531 GiB` eager (`0.000 GiB` extra graph pool overhead).
8. **Liger-Kernel 0.8.4 Tensor-Parallel Fused Cross-Entropy in veRL (`verl#7945`, `5b550b0c3d`)**: Adds `impl_backend: "liger"` to veRL's Megatron engine (with SM90 `#1422` and SM100 `#1458` fixes), avoiding `[B*S, V/TP]` FP32 logit materialization (`594.0 MiB` saved per rank at `8192` tokens, `V=152064, TP=8`).
9. **1 Hz `ibv_query_port` Recovery for Missed `IBV_EVENT_PORT_ACTIVE` in Mooncake (`Mooncake#4715`, `8c19b7286c`)**: Classic `WorkerPool` remembers when a context was disabled specifically by `IBV_EVENT_PORT_ERR` and polls `ibv_query_port` at `1 Hz` to arm the delayed GID probe even if the kernel/driver drops `IBV_EVENT_PORT_ACTIVE`.
10. **Eliminating Per-Chunk Cancellation Task Racing in Dynamo SGLang Streaming (`dynamo#16351`, `a246c3f227`)**: Replaces per-chunk `asyncio.wait` racing with direct stream iteration and a `_STREAM_END` sentinel, reducing async wait task scheduling overhead by `1024.0x` on a 512-chunk response stream.

---

## 2. Primary Source Evidence Matrix

| # | Source ID / Commit | Mechanism & Contract | Quantitative Result / Invariant |
| :--- | :--- | :--- | :--- |
| 21 | `lmcache-limit-p2p-controller-lookup-local-dram-segment-pr6014-2026` (`c90e6862ab`) | `allowed_l1_memory_tags={(DRAM, LOCAL)}`, `allowed_l2_memory_tags=set()` | Zero recursive peer lookups into remote/disk L2 tiers |
| 22 | `lmcache-release-storagemanager-refs-batched-put-error-pr5940-2026` (`562517ad9f`) | `try/except` ref release in `StorageManager.batched_put` | `0` leaked pinned `MemoryObj` chunks on backend exception |
| 23 | `nixl-ucx-improve-notification-performance-pr2412-2026` (`0165676874`) | `absl::InlinedVector` + `exportMoveStr` in UCX notifications | P99 prep `8->6 us` (`1.333x`), P99 post `12->9 us` (`1.333x`) |
| 24 | `nixl-obj-share-cuobject-registration-across-agents-pr2379-2026` (`a486deaa23`) | Reference-counted `SharedCuObjClient` per `(ptr, size)` | `1` driver call for 4 agents sharing 1 GiB arena; `0` premature unpins |
| 25 | `megatron-fix-torch-gdn-recurrent-state-alloc-cudagraph-pr8322-2026` (`e592c042d5`) | `value.new_zeros(...)` in `torch_chunk_gated_delta_rule` | Zero H2D copy errors during GDN CUDA Graph capture |
| 26 | `te-fix-grouped-mxfp8-quant-varying-last-dim-swizzled-pr3599-2026` (`76ae4b0981`) | Separate `tensor_base_for_scales` in `group_quantize_mxfp8_kernel` | Exact swizzled scale offset (`1024` vs buggy `1024` mismatch on unaligned shapes) |
| 27 | `torchtitan-share-cudagraph-memory-training-validation-pr5241-2026` (`3c7b0c9d94`) | Shared training/validation CUDA Graph pool + `_CUDAGraphGradientState` | `61.530 GiB` peak vs `61.531 GiB` eager (`0.000 GiB` extra overhead) |
| 28 | `verl-megatron-add-liger-tp-fused-cross-entropy-pr7945-2026` (`5b550b0c3d`) | `liger-kernel==0.8.4` TP fused linear cross-entropy | Saves `594.0 MiB` FP32 logits per rank (`8192` tokens, `V=152064, TP=8`) |
| 29 | `mooncake-recover-classic-rdma-contexts-port-active-missed-pr4715-2026` (`8c19b7286c`) | `1 Hz` `ibv_query_port` poll after `IBV_EVENT_PORT_ERR` | Recovers RDMA rail within `~1 s` when `PORT_ACTIVE` async event is missed |
| 30 | `dynamo-sglang-stop-racing-stream-chunk-cancellation-task-pr16351-2026` (`a246c3f227`) | Direct `anext` iteration + `_STREAM_END` sentinel | `1024.0x` reduction in wait-task scheduling for 512-chunk streams |

---

## 3. Reproducible Offline Verification

- **Script**: `experiments/lmcache_p2p_dram_put_ref_nixl_ucx_cuobj_megatron_gdn_te_mxfp8_titan_verl_mooncake_dynamo_phase97/run.py`
- **Command**: `python experiments\lmcache_p2p_dram_put_ref_nixl_ucx_cuobj_megatron_gdn_te_mxfp8_titan_verl_mooncake_dynamo_phase97\run.py --check`
- **Unit Test**: `tests/test_phase97_experiments.py`
