# Phase 93 Track 2: SGLang DeepSeek-V4.1 & Diffusion GC, vLLM KV-Offload Sync-Hit/SWA-Null/OBJ, and TensorRT-LLM

Deterministic Python 3.12 standard-library model reproducing the serving and KV-offload optimizations in Phase 93 Track 2:

1. **SGLang DeepSeek-V4.1 & Diffusion Worker (`#46530`, `#46263`, `#46220`, `#46493`)**:
   - Writes Engram hash IDs and embeddings into caller-allocated prefill graph buffers (`#46530`).
   - Restricts `flashmla_mega` default on aggregated servers to `attn_tp_size == 1` (`64` query heads per rank) while keeping `flashmla_mega` across PD disaggregation (`#46263`).
   - Orders coarse PD prefill WAR barriers after `process_batch_result` and `process_disagg_prefill_inflight_queue` (`#46220`).
   - Defers automatic cyclic Python GC during diffusion request execution with a 30s drain window (`#46493`).
2. **vLLM KV Offloading & ROCm GDN (`#61961`, `#63652`, `#63655`, `#63558`)**:
   - Loads small CPU-tier KV hits synchronously within the same step up to `sync_load_budget_bytes`, eliminating the 1-step async bubble (`#61961`).
   - Skips storing sliding-window KV chunks that contain evicted `null_block` pointers (`#63652`) and namespaces object-tier keys by `blocks_per_chunk` (`#63655`).
   - Guards ROCm AITER GDN decode fast path against `is_conv_state_dim_first()` layouts (`#63558`).
3. **TensorRT-LLM Beam-History Rank Check & Driverless `trtllm-render` (`#21027`, `#20842`)**.
