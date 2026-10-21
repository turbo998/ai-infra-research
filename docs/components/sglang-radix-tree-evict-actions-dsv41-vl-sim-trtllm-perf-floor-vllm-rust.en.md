# SGLang Unified Radix-Tree Eviction Actions & DS-V4.1 Vision MoE Simulation, TRT-LLM History Floor, and vLLM Multimodal/XPU/Rust Contracts

> **First-Hand Upstream Sources** (`2026-10-21`):
> - `sgl-project/sglang@5e79333c95c162abf079548fe40a6a2bca6c5409` (`#45439`): *[Radix Tree] Refactor custom tree core (#45439)*
> - `sgl-project/sglang@ed268dd32097ed44a83b260250defb82256602e2` (`#45628`): *Fix routing simulation for DeepSeek-V4.1 vision-biased MoE gates (#45628)*
> - `NVIDIA/TensorRT-LLM@7f0caf443ec440ffc7ca455c91ff6f7c8b0b8d67` (`#20297`): *[https://nvbugs/6925906][infra] Floor gen_only device step time perf history at metric cutover (#20297)*
> - `vllm-project/vllm@65a886c69f44cf446d6afa03315f39590ccd843e` (`#62859`): *[Bugfix] Reject multimodal inputs when chunked processing is enabled (#62859)*
> - `vllm-project/vllm@35d01660ff3644de128253950b8fcb4b7705668f` (`#62244`): *[XPU] Enable speculative decoding adaptive verification without confidence head on XPU (#62244)*
> - `vllm-project/vllm@18e8791134e8e46134f982c8263d5fa0a50548c0` (`#62823`): *[Rust Frontend] Apply response_format over structured_outputs like Python (#62823)*

---

## 1. SGLang Unified Radix-Tree Eviction Action Propagation (`#45439`) & DeepSeek-V4.1 Vision MoE Routing Simulation (`#45628`)

### 1.1 Cross-Language Python & Rust Radix-Tree Eviction Contract (`#45439`)

In hybrid prefix caches combining Full KV, Mamba state, Sliding Window Attention (`SWA`), and custom sidecar components (`C128SidecarComponent`), eviction previously tracked only `device_frees` and `host_frees`. Custom component sidecar actions emitted during device eviction, host promotion/eviction, cascade eviction (`_cascade_evict`), or `release_window_lock` were not threaded through the unified tree core. Commit `5e79333c95` (`#45439`, `+3,379 / -2,027` across 31 Python/Rust files) unifies the insert walk and eviction lifecycle:
1. **`cache_actions` Propagation Across Python & Rust**: Threads `cache_actions: list[CacheAction | ComponentAction]` through `evict_component`, `evict_device_next_node`, `drive_host_eviction`, `_cascade_evict`, and `release_window_lock` in both Python (`CustomRadixTreeCore`) and Rust (`sglang-radix-tree`).
2. **`BaseEvictionResult.__del__` Undrained-Resource Tripwire**: Enforces an explicit destructor invariant:
   ```python
   assert (
       not self.device_frees
       and not self.host_frees
       and not self.cache_actions
   ), "EvictionResult destroyed with undrained resources"
   ```
   guaranteeing that any caller failing to drain `device_frees`, `host_frees`, or `cache_actions` triggers an immediate failure rather than leaking pools or sidecar state.

### 1.2 Balanced Routing Simulation for DeepSeek-V4.1 Vision-Biased MoE Gates (`#45628`)

DeepSeek-V4.1 vision-biased MoE gates execute via `_finish_vision_topk` in `vl_routing.py`, bypassing `CustomTopK` and therefore ignoring `SGLANG_SIMULATE_UNIFORM_EXPERTS` and `SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS`. Commit `ed268dd320` (`#45628`) extracts `sglang/srt/layers/moe/topk_simulation.py` and wires it into `_finish_vision_topk` with three critical invariants:
1. **Shared-Expert Slot Preservation**: Rewrites only `num_routed_topk = top_k - num_shared_topk_slots` columns so fused shared-expert slots appended at the tail of `topk_ids` remain intact.
2. **Sequence-Sharded Token Rank Offset**: When `moe_input_scattered=True` and `attn_tp_sequence_sharded=True`, computes:
   $$\text{token\_shard\_rank} = \text{attn\_dp\_rank} \times \text{attn\_tp\_size} + \text{attn\_tp\_rank}, \quad \text{num\_token\_shards} = \text{attn\_dp\_size} \times \text{attn\_tp\_size}$$
   preventing TP ranks within an attention DP group from generating identical round-robin expert IDs.
3. **CUDA Graph Padded-Row Re-Masking**: Resets `padded_rows_masked = False` after `apply_routing_simulation` overwrites `topk_ids`, forcing `_post_process_topk_ids` to re-mask padded CUDA graph rows back to `-1`.

---

## 2. TensorRT-LLM `gen_only` Device-Step-Time History Floor (`#20297`)

When `ceil(concurrency / dp) > max_batch_size` (e.g., `concurrency=64, max_batch_size=32, osl=400`), generation runs in a full-batch wave (`~7.5 ms/iter`) followed by a short tail wave (`~2.5 ms/iter`, **`~3.0x` faster**) with nearly identical iteration counts (`~420` iterations).
- Prior to `2026-10-01`, `gen_only` picked the modal `num_generation_tokens` bucket, randomly selecting the `~2.5 ms` tail wave whenever iteration counts tied.
- After `2026-10-01`, `gen_only` locked onto `expected_gen_only_ngen` (the full-batch wave at `~7.5 ms`), but the 90-day 3-day-smoothed **P5 baseline** remained depressed at **`3.3117 ms`** due to pre-cutover tail-wave points.
- Commit `7f0caf443e` (`#20297`) floors the 5 `gen_only` step-time metrics (`d_{mean,median,std,p75,p99}_gen_worker_per_iter_device_step_time`) at `GEN_ONLY_DEVICE_STEP_TIME_HISTORY_FLOOR_MS = 1790856543000` (`2026-10-01T12:09:03Z`) and invalidates pre-floor stored `d_baseline_*` pins while preserving `d_threshold_*` overrides and `e2e` history, restoring the P5 baseline to **`7.4960 ms`**.

---

## 3. vLLM Multimodal Chunked Embedding Guard (`#62859`), XPU Spec-Decode Estimator (`#62244`) & Rust `response_format` Precedence (`#62823`)

1. **Multimodal Chunked Embedding Validation (`#62859`)**: Scans the entire batch in `PoolingServeContext._maybe_pre_process_chunked` and raises `VLLMValidationError` (`400 Bad Request`) before mutating `engine_prompts` if any item contains multimodal inputs, preventing mixed text+multimodal batches from corrupting embeddings.
2. **XPU Speculative Decoding `OnlineAcceptanceEstimator` (`#62244`)**: Enables confidence-head-free adaptive verification on XPU using a damped $O(K)$ arrowhead solver (`DAMPING_OBSERVATIONS = 20.0`) and `AttentionCGSupport.ALWAYS`.
3. **Rust OpenAI Server `response_format` Precedence (`#62823`)**: Aligns `convert_from_response_format` with Python so a non-`text` `response_format` overrides a same-discriminant `structured_outputs` constraint while retaining its auxiliary options (`whitespace_pattern`, etc.), and rejects cross-kind constraints with `400 Bad Request`.
