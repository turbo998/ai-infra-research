# Phase 93 Track 2: SGLang DeepSeek-V4.1 & Diffusion GC Isolation, vLLM KV-Offload Sync-Hit Budget/SWA Null-Block Filter, and TensorRT-LLM Driverless Render

> **中文版 / Chinese Version**: [`sglang-dsv41-vllm-kv-offload-trtllm-phase93.md`](sglang-dsv41-vllm-kv-offload-trtllm-phase93.md)  
> **Reproducible Experiment**: [`experiments/sglang_dsv41_vllm_kv_offload_trtllm_phase93/`](../../experiments/sglang_dsv41_vllm_kv_offload_trtllm_phase93/)  
> **Upstream Commits Covered (`2026-10-26`)**:
> - `sgl-project/sglang@52a6f7049f` (`#46530`): `[Refactor] dsv41: Write engram hash ids and embeddings into caller-allocated prefill graph outputs`
> - `sgl-project/sglang@60d941293e` (`#46493`): `[Perf] diffusion/worker: Move cyclic GC out of request execution`
> - `sgl-project/sglang@655c2cdd64` (`#46263`): `[Perf] [NVIDIA] deepseek-v4.1: Default aggregated servers to MegaAttention only when each rank holds 64 query heads`
> - `sgl-project/sglang@d7f832d30a` (`#46220`): `[Perf] pd: Apply the prefill WAR barrier after the previous result is processed`
> - `vllm-project/vllm@94b35f3780` (`#63558`): `[Bugfix][ROCm] Keep the AITER GDN decode fast path off for the DS conv-state layout`
> - `vllm-project/vllm@8c6e770a1c` (`#63655`): `[Bugfix][KV Offload][OBJ] Namespace object keys by blocks_per_chunk`
> - `vllm-project/vllm@1d3f05aadb` (`#63652`): `[Bugfix][KV Offload] Do not store sliding-window chunks with null blocks`
> - `vllm-project/vllm@fe5fe8ba5f` (`#61961`): `[KV Offload] Load small CPU-tier hits synchronously within the step`
> - `NVIDIA/TensorRT-LLM@e7d3441e77` (`#21027`): `[None][fix] Use decoder mapping for beam-history rank checks`
> - `NVIDIA/TensorRT-LLM@cefbde86e8` (`#20842`): `[TRTLLM-17241][fix] Start trtllm-render on a host with no GPU driver`

---

## 1. Architectural Overview

In disaggregated serving and multi-tier KV offloading pipelines, subtle scheduler/worker contract mismatches degrade both latency and correctness: **(1) forcing small CPU-tier KV cache hits through an asynchronous next-step barrier adds a full 10–15 ms step bubble to TTFT**, **(2) storing evicted sliding-window `null_block` entries or omitting `blocks_per_chunk` from object-store keys corrupts cached KV state**, and **(3) defaulting aggregated servers to `flashmla_mega` when tensor parallelism splits query heads below 64 per rank regresses kernel throughput**. This report analyzes 10 upstream commits across SGLang, vLLM, and TensorRT-LLM.

---

## 2. SGLang DeepSeek-V4.1 Graph Allocation, MegaAttention Gating & Diffusion GC Deferral

1. **Caller-Allocated Engram Graph Outputs (`#46530`, `52a6f7049f`)**: Adds `hasher.new_hash_ids(input_ids)` and updates `deepseek_v4_engram_hash_ids(hasher, input_ids, out)` to copy into caller-allocated buffers across piecewise breakable CUDA graph boundaries instead of allocating fresh tensors inside eager graph breaks.
2. **MegaAttention 64-Query-Head Gate on Aggregated Servers (`#46263`, `655c2cdd64`)**: `_attention_is_tensor_parallel` checks whether `derive_attn_tp_size(tp_size, attn_cp_size, attn_dp_size) > 1` when `enable_prefill_cp` is off. Outside PD disaggregation, SGLang now defaults `--dsv4-attn-backend auto` to `flashmla_mega` only when each rank holds all **64 query heads** (`attn_tp_size == 1`), falling back to `flashmla` when attention is tensor-parallel.
3. **PD Disaggregation Coarse WAR Barrier Ordering (`#46220`, `d7f832d30a`)**: Defers the coarse whole-forward WAR barrier until after `process_batch_result` and `process_disagg_prefill_inflight_queue` (`launch -> process_result -> inflight -> barrier`), overlapping host result processing with device execution.
4. **Diffusion Worker Cyclic GC Deferral (`#46493`, `60d941293e`)**: Introduces `RequestGarbageCollector`, settling startup/warmup GC debt eagerly and deferring automatic cyclic GC during active request forward/reply execution with a bounded `MAX_DEFER_SECONDS = 30.0` drain window under `--async-output-save`, eliminating `38.5 ms` full-heap GC pauses (`8.40%` step P99 jitter reduction).

---

## 3. vLLM KV Offloading Synchronous Budget, SWA Null-Block Filtering & ROCm GDN Guard

1. **Synchronous Small CPU-Tier KV Hits (`#61961`, `fe5fe8ba5f`)**: Introduces `sync_load_budget_bytes`. When CPU-tier KV cache hits fit within the per-step per-GPU byte budget and leave tokens to compute, the scheduler dispatches `TransferJob(..., sync=True)` within the same engine step. For a `16 MiB` step budget, `4 MiB` + `8 MiB` hits incur only `0.25 ms` inline PCIe Gen5 DMA overhead while saving **11.75 ms** of next-step scheduling latency.
2. **Sliding-Window Null-Block Store Filter (`#63652`, `1d3f05aadb`)**: Prevents the KV offloading connector from storing sliding-window chunks whose blocks have already been evicted to `null_block` outside the $2 \times \text{block\_size}$ window.
3. **Object-Tier Key Namespacing by `blocks_per_chunk` (`#63655`, `8c6e770a1c`)**: Passes `blocks_per_file=offloading_spec.blocks_per_chunk` to `FileMapper.from_offloading_spec`, isolating object keys across different chunk granularities.
4. **ROCm AITER GDN Decode Fast-Path Guard (`#63558`, `94b35f3780`)**: Disables `_forward_core_decode_aiter` when `is_conv_state_dim_first()` is active for DeepSeek conv-state layouts.

---

## 4. TensorRT-LLM Beam-History Rank Check & Driverless CPU `trtllm-render` (`#21027`, `#20842`)

- **Decoder Mapping Fix (`#21027`, `e7d3441e77`)**: Fixes `DecoderRunner._start_beam_histories` to check `self._mapping.is_last_pp_rank()` instead of the non-existent `self.mapping`, ensuring beam history gathers queue ahead of the forward pass on the last pipeline stage.
- **Driverless `trtllm-render` Entry Point (`#20842`, `cefbde86e8`)**: Moves `trtllm-render` to a standalone script that preloads the CUDA toolkit's `libcuda.so` stub from `$CUDA_HOME` or `/usr/local/cuda*` when no GPU driver is present, enabling CPU-only preprocessing containers with `render_fingerprint.digest` verification.
