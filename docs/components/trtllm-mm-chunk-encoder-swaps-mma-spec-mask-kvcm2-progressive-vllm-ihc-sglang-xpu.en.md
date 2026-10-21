# Deep Dive: TensorRT-LLM Chunk-Scoped Multimodal Scheduling, Blackwell `SwapsMmaAb` Speculative Mask Sizing & KVCM2 Progressive Host Commit, vLLM `libgomp` Fork Pause & ROCm HY-V4 iHC Fusion, and SGLang Intel XPU Disaggregated KV

> **Research Phase**: Phase 46 (`priority: 55`) · **Question ID**: `q-phase46-trtllm-mm-chunk-spec-mask-kvcm2-vllm-ihc-sglang-xpu` · **Deterministic Script**: [`experiments/trtllm_mm_chunk_spec_mask_kvcm2_vllm_ihc_sglang_xpu/run.py`](../../experiments/trtllm_mm_chunk_spec_mask_kvcm2_vllm_ihc_sglang_xpu/run.py)

---

## 1. Executive Summary & First-Hand Upstream Commits

Across multimodal long-context serving, Blackwell speculative tree decoding (Eagle/Medusa), hierarchical KV cache management (KVCM2), and heterogeneous GPU backends (ROCm / Intel XPU), recent commits in `NVIDIA/TensorRT-LLM`, `vllm-project/vllm`, and `sgl-project/sglang` resolve five production bottlenecks:

1. **Whole-Request Multimodal Encoder Head-of-Line Blocking (`TensorRT-LLM#19884`)**: Previously, `DisaggScheduleSimulator` and `MultimodalRuntime` required all multimodal items of a request to fit within `encoder_max_num_tokens` and `EncoderCacheManager` capacity at once, starving multi-image/video requests even under chunked prefill and KV prefix hits.
2. **Blackwell Speculative Decoding Custom-Mask Buffer Overflow under `SwapsMmaAb` (`TensorRT-LLM#20223`)**: Python allocated `spec_decoding_bl_tree_mask` using only the `KeepsMmaAb` formula (`stepQ=128, stepKv=256`). When the FMHA dispatcher selected a `SwapsMmaAb` generation kernel (`tileSizeQ` padded to `32` and tiled per query token), the kernel wrote up to **`4.0x`** more `uint32` mask words than allocated.
3. **KVCM2 Synchronous Host-Tier Startup Pinning & `mremap` Base Relocation (`TensorRT-LLM#19650`)**: `HostMem` previously committed and pinned its entire host quota synchronously at initialization or grew via `mremap` (risking pointer invalidation).
4. **vLLM Post-`fork` `libgomp` Deadlock (`vllm#62884`) & ROCm HY-V4 Cross-Layer iHC Bandwidth (`vllm#58509`)**: Forking worker processes after any parent PyTorch CPU OpenMP region left child processes deadlocked on their first OpenMP region; on ROCm, HY-V4 executed `ihc_post`, `ihc_pre`, and `RMSNorm` as 3 separate kernels over `[T, 4, d]` hyper-connection tensors.
5. **SGLang Intel XPU Disaggregated KV Page Faults (`sglang#40861`) & FlashInfer CUTLASS MoE Bucket Mismatch (`sglang#43363`)**: Intel XPU places device memory in kernel-space high virtual addresses (`ptr & (1 << 63) != 0`) and corrupts non-8B-aligned MLA slices when packed as `int64`; meanwhile prefill workers over-tuned MoE buckets without an `8192`-token cap.

| Commit | Date | Architectural Mechanism | Measured / Verified Impact |
|---|---|---|---|
| `TensorRT-LLM@b47eb8fe6b` (`#19884`) | 2026-10-21 | `ContextChunk(start, end)` resolves KV prefix reuse first, budgets multimodal items per chunk, and adds `TensorLRUCache.releasable_bytes` | Schedules 4-image (`4096`-token) requests across `2048`-token chunks and releases consumed items after each chunk |
| `TensorRT-LLM@85b1b8323b` (`#20223`) | 2026-10-19 | `getCustomMaskCapacityUpperBound` maximizes across `KeepsMmaAb` and `SwapsMmaAb` shapes and checks `customMaskCapacity` | Prevents **`4.0x`** buffer overflow (`2,097,152` vs `524,288` `uint32` words at `B=64, Q=64, H_q/kv=8`) |
| `TensorRT-LLM@0cd9d0b541` (`#19650`) | 2026-10-19 | KVCM2 `HostMem` reserves `MAP_NORESERVE` upfront and commits progressively in `2 MiB -> 32 MiB` units on a background worker | Cuts synchronous startup commit at `64 GiB` quota from `64 GiB` to `32 MiB` (**`2048x`** reduction) without `mremap` |
| `vllm@40210d8317` (`#62884`) | 2026-10-21 | `_release_libgomp_threads()` invokes `omp_pause_resource_all(_OMP_PAUSE_HARD=2)` on loaded `libgomp` libraries before `fork` | Eliminates post-fork OpenMP worker deadlocks when spawning engine core processes |
| `vllm@833349105b` (`#58509`) | 2026-10-21 | ROCm HY-V4 fused `triton_ihc_post_pre_rms_norm` kernel + AITER MXFP8 `MoEActivation.SILU` support | Cuts kernel launches from `3 -> 1` and reduces per-token HBM traffic from `128 KiB -> 80 KiB` (**`1.60x`** reduction) |
| `sglang@8e3e2980db` (`#40861`), `f8fd41c36e` (`#43363`), `2c9d919071` (`#45437`) | 2026-10-21 | Intel XPU `ptr & (1 << 63)` detection + host staging + per-token MLA RDMA; MoE prefill `min(T * dp, 8192)`; `SGLANG_ZMQ_SOURCE_IP` | Enables lossless XPU disaggregated KV transfer, bounds CUTLASS MoE tuning buckets, and pins ZMQ source NICs |

---

## 2. TensorRT-LLM: Chunk-Scoped Multimodal Budgeting & `SwapsMmaAb` Mask Capacity

### 2.1 Prefix-Aware `ContextChunk` Multimodal Scheduling (`#19884`)

1. **Resolve KV Prefix Reuse Before Multimodal Budgeting**: `DisaggScheduleSimulator` queries the KV cache radix tree before selecting multimodal items, skipping embeddings already covered by reused prefix blocks;
2. **Per-Chunk Window Accounting**: `get_mm_items_for_chunk` charges only unencoded items overlapping `[chunk_start, chunk_end)` against `encoder_max_num_tokens` and frees embeddings via `free_chunk_mm_data` as soon as a chunk moves past their end offset;
3. **Reference-Counted `releasable_bytes(keys)`**: `TensorLRUCache.releasable_bytes` counts occurrences of each key in the candidate set (`counts = Counter(keys)`) and counts `entry.nbytes` only when `0 < entry.reference_count <= count`, preventing overestimation when another in-flight request pins the same multimodal hash.

### 2.2 Blackwell `SwapsMmaAb` vs `KeepsMmaAb` Custom-Mask Sizing (`#20223`)

In `prepareCustomMask.cu`, `computeSwapsMmaAbCustomMaskSize` rounds `tileSizeQRaw` $\in \{8, 16, 32\}$ up to `tileSizeQ = 32` and tiles `numHeadsQPerKv` per query token (`numTilesQ = seqLenQ * ceilDiv(numHeadsQPerKv, 32 * numInstsQ)`). For `maxBatchSize = 64, maxSeqLenQ = 64, numHeadsQPerKv = 8`:
- Legacy `KeepsMmaAb`-only sizing (`stepQ = 128, stepKv = 256`) allocates **`524,288` `uint32` elements (`2 MiB`)**;
- `SwapsMmaAb` (`tileSizeQ = 8/16/32, stepKv = 128`) requires **`2,097,152` `uint32` elements (`8 MiB`, `4.0x` larger)**.
`getCustomMaskCapacityUpperBound` takes the maximum over all candidate generation kernel shapes and enforces `required <= runnerParams.customMaskCapacity` at launch.

---

## 3. KVCM2 Progressive Host Commit, vLLM `libgomp` Fork Pause & SGLang Intel XPU

1. **KVCM2 Progressive `HostMem` (`#19650`)**: Reserves the full `max(maxQuota, alignedSize)` virtual range upfront with `MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE` and commits physical pages in units of $\text{roundUp}(\max(\text{tierUnit}, \text{slotSize}), \text{slotSize})$ where $\text{tierUnit} = (2\text{ MiB}) \ll \min(4, \lfloor \log_2(Q / 1\text{ GiB}) \rfloor)$. At `64 GiB` quota, synchronous startup work drops from `64 GiB` to a single `32 MiB` unit (**`2048x`** reduction).
2. **vLLM `libgomp` Fork Safety (`#62884`) & ROCm HY-V4 iHC Fusion (`#58509`)**: Calling `omp_pause_resource_all(_OMP_PAUSE_HARD=2)` before `os.fork()` joins `libgomp` worker threads so child engine processes never deadlock on inherited OpenMP state; `triton_ihc_post_pre_rms_norm` fuses channel scatter, cross-channel RMSNorm, sigmoid gating, and output RMSNorm into one FP32 Triton kernel (**`1.60x`** HBM traffic reduction).
3. **SGLang Intel XPU Disaggregated KV (`#40861`, `#43363`, `#45437`)**: Detects XPU kernel-space addresses via `bool(ptr & (1 << 63))`, stages transfers through pinned host buffers with `_await_handles()` synchronization, disables `int64` `copy_mla_rows_into_pack` on XPU, caps prefill CUTLASS MoE tuning at `min(prefill_tokens * attn_dp_size, 8192)`, and supports `SGLANG_ZMQ_SOURCE_IP` (`tcp://<src>:0;<dst>`).
