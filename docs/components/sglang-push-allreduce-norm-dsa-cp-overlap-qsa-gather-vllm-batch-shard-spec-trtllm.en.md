# SGLang Push-Plane TP All-Reduce + Residual + RMSNorm Fusion, DSA CP KV Gather Overlap, QSA Single-Launch Gather, EAGLE HiCache Bigram Boundary, and vLLM Batch-Sharded Adaptive Verification & TRT-LLM MiniMax-M3 VL

## 1. Engineering Problem Definition & Architectural Tensions

In production serving for DeepSeek-V3/V4, Qwen3-MoE, and MiniMax-M3 VL, tail latency and numerical correctness depend on layer-boundary fusion, sparse attention (DSA/QSA) indexer preparation, and speculative/offload boundary invariants:

1. **Redundant HBM Round-Trips at Layer Boundaries** (`sgl-project/sglang@d558ac2f3e7a`, `#46137`): Launching separate kernels for MoE finalize, CustomAllReduce, residual add, and RMSNorm at FFN inputs, next-layer attention inputs, and `final_norm` incurs 3–4 HBM round-trips per boundary.
2. **Serial Blocking Between Interleave CP KV All-Gather and DSA Indexer** (`sgl-project/sglang@5971f47bc8ea`, `#46179`): During long-context extend batches under Context Parallelism, waiting for CP KV gather before running the DSA top-k indexer wastes dual-stream overlap opportunities.
3. **Kernel Launch Storm in QSA Prefill Indexer Gather** (`sgl-project/sglang@dba22e9b68dd`, `#45752`): Looping over $N$ sequences with `compressed_buffer.index_select(0, compressed_locs)` followed by `torch.cat` issues $N+1$ launches per layer.
4. **Missing EAGLE Bigram Boundary Token in PD Decode HiCache Restore** (`sgl-project/sglang@0ff67ee72aaf`, `#45076`): An EAGLE radix tree keys $N$ bigrams with $N+1$ token IDs; bounding `match_kv_cache` by `decode_prefix_len` truncates the boundary token.
5. **vLLM V1 Batch-Sharded Sampling vs. GPU Adaptive Verification Conflict & MRV2/Offload Edge Cases** (`vllm-project/vllm@7050ef881ff0` `#58466`, `fc2b6ab45ed3` `#62698`, `2c19f6793e98` `#63372`, `65bc698356b4` `#63266`, `ceb01b6ddf6e` `#61976`) and **TRT-LLM MiniMax-M3 VL Prompt Double-Templating & Chunked MM Slicing** (`NVIDIA/TensorRT-LLM@a0cee56d3a6e`, `#20626`).

---

## 2. First-Hand Source Code & Mechanism Audit

### 2.1 SGLang Push-Plane TP All-Reduce + Residual + RMSNorm Fusion (`#46137`)

Auditing `python/sglang/kernels/jit/csrc/distributed/all_reduce_fusion.cuh` and `python/sglang/kernels/ops/communication/all_reduce_fusion.py`:

- **`RowClusterTrait` Geometry**: Each row of width `kHiddenDim` contains `kRowVecs = kHiddenDim / 8` 16-byte vectors. With `kVecs` vectors per thread and `kClusterSize` CTAs per cluster, `kBlockSize = kRowVecs / (kClusterSize * kVecs)`.
- **One-Shot vs. Two-Shot Policy (`H = 6144`, TP4 on B300 / TP8 on H200)**:
  - Thresholds: `TWO_SHOT_MIN_TOKENS = {4: 48, 8: 8}`.
  - Below threshold: One-Shot (`vecs_per_thread = 1, cluster_size = 4`).
  - At or above threshold: Two-Shot selects occupancy-tuned `(max_rows, vecs_per_thread, cluster_size)` from `_TWO_SHOT_GEOMETRY`:
    - Plain (`top_k == 0`): `((272, 1, 4), (None, 2, 3))` (`192` or `128` threads/block).
    - Deferred MoE Finalize (`top_k > 0`): `((208, 1, 4), (232, 1, 3), (None, 2, 3))` (`192`, `256`, or `128` threads/block).
  - Launches `dim3(num_tokens + 1, kClusterSize)` where `blockIdx.x == num_tokens` is the **bumper cluster** that flips leftover push counters `[num_tokens, num_push_counters)` for CUDA Graph safety.

### 2.2 SGLang DSA CP Overlap (`#46179`), QSA Single-Launch Gather (`#45752`), and HiCache EAGLE Fix (`#45076`)

- **DSA Interleave CP Overlap (`interleave.py`, `forward_mla.py`)**: `DSAInterleaveCPIndexerOverlap` overlaps CP KV cache gather with `self.indexer(...)` during `is_interleave_extend`, reducing serial latency $T_{\text{kv}} + T_{\text{idx}}$ to $\max(T_{\text{kv}}, T_{\text{idx}}) + \epsilon_{\text{sync}}$ (**`1.743x`** modeled speedup at 8K–128K).
- **QSA Single-Launch Triton Gather (`qsa/kernel.py`)**: `_gather_qsa_compressed_keys_kernel` gathers all sequences' complete compressed keys (`first // RATIO`) in a single 2D launch `(ceil(max_blocks / BLOCK_B), num_sequences)` copying raw bytes (`ROW = heads * dim * element_size()`), cutting $N+1$ launches to **1** (**`17.45x`** modeled speedup at 128 sequences).
- **HiCache EAGLE Bigram Boundary (`decode_hicache_mixin.py`)**: Adds `_bigram_boundary_tokens(tree_cache) = int(tree_cache.tree_core.is_eagle)` to `pm.decode_prefix_len` during `_try_hicache_queue_load_back`.

### 2.3 vLLM V1 Scheduling/Sampling/Offload & TRT-LLM MiniMax-M3 VL Fixes

- **vLLM Batch-Sharded Sampling + Adaptive Verification (`#58466`)**: Copies GPU-decided `AdmittedDraftCounts` to the host so `BatchSharder` shards the exact admitted logit count rather than the static upper bound (**`16.0x`** per-rank sampling reduction at `TP=8`).
- **vLLM 2D Chunked Staged Writes (`#62698`)**: Parallelizes `_apply_write_kernel` over `(n, ceil(max_len / 1024))` (**`16.0x`** wave speedup for 32 requests × 64K elements).
- **vLLM DBO Microbatch Routed-Expert Offset (`#63372`)**: Slices `self.device_buffer[offset : offset + len(local_topk_ids), layer_id]` using `dbo_current_ubatch_token_offset()`.
- **vLLM Chunked-Local KV Offload Alignment (`#63266`)**: Rejects CPU suffix cache hits when `num_computed_tokens % spec.attention_chunk_size != 0`.
- **vLLM Priority Self-Preemption (`#61976`)**: Uses `continue` instead of `break` when the lowest-priority running request preempts itself so higher-priority requests after it can claim its freed blocks.
- **TRT-LLM MiniMax-M3 VL (`#20626`)**: Adds `chat_template_applied` and `supports_token_id_mm_expansion = True`, and slices multimodal embeddings via `find_input_mm_embeds` under KV reuse and chunked prefill.

---

## 3. Reproducible CPU/Offline Verification

```powershell
python experiments\sglang_push_allreduce_norm_dsa_cp_overlap_qsa_gather_vllm_batch_shard_spec_trtllm\run.py --check
```

| Engine / Feature | Scenario | Before | After | Key Metric / Gain | Evidence Class |
|---|---|---:|---:|---:|---|
| SGLang Push-Plane AR+Norm (`#46137`) | `H=6144, WS=8, T=224`, w/ Finalize | 4 separate kernels | 1 `two_shot` `(1, 3)` (`256` thr) | `225` Clusters (+1 bumper) | `implementation` + `own-offline` |
| SGLang DSA CP Overlap (`#46179`) | `seq_len=131072, CP=4` | `2686.98 us` (serial) | `1541.49 us` (overlapped) | **`1.7431x`** latency reduction | `implementation` + `own-offline` |
| SGLang QSA Gather (`#45752`) | `128` prefill sequences | `129` launches (`492.9 us`) | **1** Triton launch (`28.24 us`) | **`17.454x`** gather speedup | `implementation` + `own-offline` |
| vLLM Staged Writes (`#62698`) | `32` reqs × `65536` elems | 1D Grid `(32,)` (`64` waves) | 2D Grid `(32, 64)` (`4` waves) | **`16.0x`** wave reduction | `implementation` + `own-offline` |
| vLLM Adaptive Batch-Shard (`#58466`) | `TP=8, B=32`, max 5 vs admitted 2 drafts | `192` logits (blocked) | **`12` logits/rank** | **`16.0x`** per-rank work cut | `implementation` + `own-offline` |

---

## 4. Architect Exercises & Falsifiable Questions

1. **Geometry Exercise**: In SGLang `#46137` with `kHiddenDim = 6144`, `world_size = 8`, `tokens = 220`, and `has_finalize = True`, determine `(vecs_per_thread, cluster_size)` and `kBlockSize`.
   - **Solution**: Since $220 \ge 8$ and $208 < 220 \le 232$, `_TWO_SHOT_GEOMETRY[True]` selects `(1, 3)`. With `kRowVecs = 6144 / 8 = 768`, `kBlockSize = 768 / (3 * 1) = 256` threads (8 warps).
2. **Falsifiable Question**: Why must `get_num_new_matched_tokens` reject a CPU KV offload suffix hit when `num_computed_tokens = 256` and `attention_chunk_size = 512`?
   - **Verification**: Chunked-local attention windows reset every 512 absolute token positions; loading a suffix cached at a different modulo offset corrupts local attention masks unless `num_computed_tokens % 512 == 0`.
