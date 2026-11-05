# LLM Inference & Serving Stack: TensorRT-LLM Single-Rank CP Bypass & 64-Bit Causal Conv1d Offsets, vLLM EPLB-Aware EP Weight Filter & Zero-Reparse Batch Writer, SGLang Qwen3 Fused QK-Norm+RoPE & Bulk DFlash Decode Prep, ROCm AITER 3D Split-KV Mask & Dynamo Retained Prepared Router Rows (Phase 183)

> 🌐 **Language / 语言切换**: **[中文](inference-trtllm-helix-cp-conv1d-64b-vllm-moe-restorer-eplb-sglang-qwen3-qknorm-dflash-dynamo-router-phase183.md)** | **[English](inference-trtllm-helix-cp-conv1d-64b-vllm-moe-restorer-eplb-sglang-qwen3-qknorm-dflash-dynamo-router-phase183.en.md)**

---

## 1. Core Architectural Problems & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 183 (`priority: 192`) audits **10 upstream mainline commits** across TensorRT-LLM, vLLM, SGLang, ROCm AITER, and NVIDIA Dynamo:

1. **ROCm AITER 3D Split-KV Tail-Row Stat Masking & TensorRT-LLM Single-Rank CP Bypass / 64-Bit Causal Conv1d Offsets**:
   - `656e98e68b49` (#7472) fixes a non-deterministic segment-stat race in AITER's Triton `unified_attention` 3D split-KV kernel when `BLOCK_Q * num_queries_per_kv < BLOCK_M` (e.g., row 15 of 16 for `GQA=5, BLOCK_Q=3, BLOCK_M=16`): applying `query_mask_0` to `segm_max` and `segm_expsum` stores eliminates concurrent `(M=0, L=0)` vs `(M=-inf, L=1)` overwrites and removes `~1e-3` run-to-run output drift (**`0.0` drift** after fix).
   - `1399e996099f` (#21192) ignores `cp_config["cp_type"]` in TensorRT-LLM's PyTorch executor when `context_parallel_size == 1` (preventing `NotImplementedError` on default `ULYSSES` or `RING`) and requires `HELIX` when `cp_size > 1`; `eaa88a0fed55` (#21684) promotes `causal_conv1d` `conv_state` slot offset arithmetic to 64-bit (`int64_t` / `size_t`), preventing 32-bit signed overflow on hybrid Mamba/conv-state pools exceeding $2^{32}$ elements.
2. **vLLM EPLB-Aware EP Weight Filter, Shared Routed-Expert Row Restorer & Zero-Reparse `run-batch` JSON Writer**:
   - `2f8c840032c6` (#64187) enables `--enable-ep-weight-filter` alongside the Expert Parallel Load Balancer (`--enable-eplb`) by deriving each rank's required experts (including redundant replicas) from EPLB's initial physical placement. On DeepSeek-R1-0528-NVFP4-v2 (`DEP=8` + `32` redundant experts), per-rank checkpoint reads drop from **`384.9 GiB` to `105.2 GiB` (`3.659x` reduction / `-72.67%`)**, warm page-cache load time drops from `43.6 s` to `39.8 s`, and GSM8K accuracy matches baseline (`0.9636` vs `0.9621`, `97.4%` exact question agreement).
   - `463e23b1dee0` (#64689) allows MoE backends to bind a shared `restore_routed_experts_rows(topk_ids, ctx)` callback in `routed_experts_capturer.py` once across all captured layers, preserving microbatch (`ubatch`) and DeepSeek-V4.1 replay token layouts without replacing `capture()`.
   - `14d147a7a05d` (#65867) writes buffered `/v1/embeddings`, `/score`, and `/rerank` JSON response bodies directly in `vllm run-batch` without parsing and re-serializing them, cutting per-line event-loop overhead by **`4.765x`** for 128-dim embeddings (`85.3 -> 17.9 us`) and **`6.398x`** for 512-dim embeddings (`188.1 -> 29.4 us`), and reducing end-to-end wall time on 1x H100 across `864,891` inputs by **`12.5%`** (128-dim) and **`18.0%`** (Qwen3-Embedding-0.6B, 512-dim).
3. **SGLang Qwen3 Contiguous Fused QK-Norm+RoPE, Bulk DFlash Decode Prep & AITER `page_size=1` Graph Capture Fix**:
   - `c07ec416dabf` (#47785) fuses Qwen3 prefill QK RMSNorm and RoPE with direct contiguous Q writeback; `e32f88502e8c` (#47678) replaces per-request Python `__setitem__` loops in `DFlashDraftInputV2.prepare_for_decode` with vectorized bulk fills, removing **`~0.4 ms`** of the `1.0–1.4 ms` pre-agreement host step time on DeepSeek-V4.1-Flash GB300 disaggregated `DEP=16` (`bs=40`/rank); `89b264138157` (#46928) copies the unified-attention page table into `cuda_graph_page_table` at `page_size=1` on AMD AITER, fixing silent garbage decode output (`GSM8K 0.00 -> full baseline`).
4. **NVIDIA Dynamo Retained Prepared Router Candidate Rows & Hybrid Conductor LPX Sidecar Gate**:
   - `4f7153790d4e` (#18009) keeps caller-prepared candidate rows in the Dynamo scheduler actor when some ranks exceed busy reject thresholds (`active_decode_blocks_threshold` / `active_prefill_tokens_threshold`), dropping only workers whose every rank is full instead of rebuilding all rows on the actor. Across **`2,048` workers** on the `agentx` workload, saturated throughput rises from **`7,172` to `10,013 req/s` (`1.396x` / `+39.6%`)**, actor CPU drops from `437` to **`314 kcycles/booking` (`-28.15%`)**, and admission p99 latency at `4.6k req/s` drops from **`2,764 ms` to `2.1 ms` (`1,316.2x` speedup)**; `7e9dc291b12a` (#18069) rejects hybrid LPX conductors lacking a `runtime` LPX sidecar at operator reconcile time.

---

## 2. End-to-End Dataflow & Architecture Topology (Pillar 1)

```
+--------------------------------------------------------------------------------------------------------+
| Phase 183 LLM Inference & Serving Stack: TRT-LLM, vLLM, SGLang, ROCm AITER & NVIDIA Dynamo (2,048 Wkr) |
+--------------------------------------------------------------------------------------------------------+
  [AITER 3D Split-KV Mask + TensorRT-LLM CP Gate & 64-Bit Conv1d (#7472, #21192, #21684)]
    |-- AITER 3D Split-KV (#7472)    --> Mask Leftover Rows (3*5=15 < 16) in segm_max/expsum (0.0 Drift)
    |-- TRT-LLM CP Gate (#21192)     --> Bypass cp_type at cp_size=1 + Enforce HELIX for cp_size>1
    +-- TRT-LLM Conv1d 64b (#21684)  --> 64-Bit int64_t Slot Offsets for >2^32 Element Hybrid State Pools
                                      |
                                      v
  [vLLM EPLB Weight Filter, Shared MoE Restorer & Zero-Reparse run-batch (#64187, #64689, #65867)]
    |-- EPLB Weight Filter (#64187)  --> DeepSeek-R1 DEP8+32 Redundant: 384.9 -> 105.2 GiB/Rank (-72.67%)
    |-- Shared MoE Restorer (#64689) --> Unified restore_routed_experts_rows Across Microbatch/Replay
    +-- Direct run-batch JSON (#65867)--> 6.398x Event-Loop Speedup (512d) & -18.0% H100 Batch Wall Time
                                      |
                                      v
  [SGLang Qwen3 QK-Norm+RoPE, Bulk DFlash Prep & Dynamo 2,048-Worker Router (#47785, #47678, #46928, #18009)]
    |-- SGLang Qwen3 & DFlash (#47678)--> Fused Contiguous QK-Norm+RoPE + -0.4 ms/step GB300 DEP16 Host Prep
    |-- SGLang AITER Graph (#46928)  --> Copy cuda_graph_page_table at page_size=1 (GSM8K 0.00 -> Full)
    +-- Dynamo Retained Rows (#18009)--> 10,013 vs 7,172 req/s (+39.6%) & Admission p99 2,764 -> 2.1 ms (1316x)
```

---

## 3. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/trtllm_helix_cp_conv1d_64b_vllm_moe_restorer_eplb_sglang_qwen3_qknorm_dflash_dynamo_router_phase183/run.py`](../../experiments/trtllm_helix_cp_conv1d_64b_vllm_moe_restorer_eplb_sglang_qwen3_qknorm_dflash_dynamo_router_phase183/run.py) `--check`:
- `aiter_split_kv_valid_rows = 15`, `aiter_split_kv_block_m = 16`, `aiter_split_kv_masked_drift = 0.0`
- `trtllm_cp_single_rank_bypass_size = 1`, `trtllm_conv1d_offset_bits = 64`
- `vllm_eplb_weight_load_reduction_ratio = 3.659` (`384.9 -> 105.2 GiB/rank`, **`72.67%` savings**)
- `vllm_run_batch_128d_speedup = 4.765`, `vllm_run_batch_512d_speedup = 6.398`, `vllm_run_batch_qwen3_emb_walltime_saving_pct = 18.0`
- `sglang_dflash_prep_saved_ms = 0.4` (`DEP=16, bs=40`), `sglang_aiter_cudagraph_fixed_page_size = 1`
- `dynamo_router_workers = 2048`, `dynamo_router_throughput_speedup = 1.396` (`7,172 -> 10,013 req/s`), `dynamo_router_kcycles_reduction_pct = 28.15`, `dynamo_router_p99_speedup = 1316.2` (`2,764.0 -> 2.1 ms`)

---

## 4. Enterprise Business Scenario Playbook (Pillar 4)

| Scenario Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **2,048-Worker Agentic Disaggregated Serving Routing** | Retain caller-prepared candidate rows past busy reject thresholds in Dynamo (`#18009`), boosting saturated throughput by `39.6%` (`10,013 req/s`) and cutting admission p99 at `4.6k req/s` from `2,764 ms` to `2.1 ms` (`1,316x`). | Discarding prepared candidate rows and rebuilding eligibility across all 2,048 workers on the scheduler actor whenever a single rank crosses a busy threshold. |
| **DeepSeek-R1 / V4.1 Large-EP + EPLB Serving & GB300 DFlash Speculative Decode** | Combine `--enable-ep-weight-filter` with `--enable-eplb` in vLLM (`#64187`, cutting per-rank checkpoint load from `384.9 GiB` to `105.2 GiB`) and enable bulk decode-prep length fill in SGLang GB300 `DEP=16` (`#47678`, `-0.4 ms/step`). | Disabling the EP weight filter whenever EPLB is active (forces every rank to read all 256+ expert weights, inflating cold-start I/O by `3.66x`). |
| **High-Throughput Batch Embeddings & AMD AITER Unified Attention** | Use zero-reparse `run-batch` JSON writing in vLLM (`#65867`, `-18.0%` wall time on Qwen3-Embedding-0.6B) and AITER 3D split-KV tail masking (`#7472`) + `page_size=1` graph page-table sync (`#46928`). | Running dense models with `SGLANG_USE_AITER_UNIFIED_ATTN=1` at `page_size=1` under HIP/CUDA Graph without syncing `cuda_graph_page_table` (produces garbage decode after the first token). |
