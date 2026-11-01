# Inference Systems: vLLM Per-Layer KV Dtype & Fused MoE Pad Skip, SGLang DSv4.1 Wave Rounding & PD Linker Dedup, and TRT-LLM GB300 Qwen3.5 (Phase 179)

> 🌐 **Language / 语言切换**: **[中文](inference-vllm-kv-dtype-moe-pad-sglang-dsv41-wave-pd-linker-phase179.md)** | **[English](inference-vllm-kv-dtype-moe-pad-sglang-dsv41-wave-pd-linker-phase179.en.md)**

---

## 1. Core Architectural Questions & Primary Source Evidence (Pillar 0 & Pillar 1)

Phase 179 (`priority: 188`) audits **10 primary upstream commits** across vLLM, SGLang, and TensorRT-LLM covering heterogeneous KV cache dtypes, MoE routing, sparse indexers, and Prefill-Decode (PD) disaggregation:

1. **vLLM Per-Layer KV Dtype Resolution, Fused MoE Padding Skip, DSv3.2 Shared PP Top-K & Qwen3.5 B200 NVFP4 Preset**:
   - `8530a778f78d` (#56155) resolves per-layer `kv_cache_dtype` directly inside the vLLM V1 attention runtime (`vllm/v1/attention/backend.py`, `gpu_model_runner.py`), enabling hybrid 56-FP8 + 8-BF16 KV caches that cut KV memory footprint by **`43.75%`** compared to uniform BF16.
   - `2a4792ddcc4b` (#64955) fuses padding-token skipping directly into grouped and Muse prepared MoE routing (`fused_moe.py`), accelerating MoE routing by **`1.21x`** (`29.04 us -> 24.00 us`).
   - `63d3d39c8472` (#59961) carries the shared DeepSeek-V3.2 indexer `top-k` tensor across pipeline-parallel (PP) stages (`0` cross-stage mismatches), and `e0d00e4b8eb0` (#64607) adds the B200 `NVFP4 TP2` agentic preset for `Qwen3.5-397B-A17B` (**`1.34x`** throughput speedup).
2. **SGLang DeepSeek-V4.1 Dense Indexer Wave Rounding, AMD Qwen3-Next Triton Router & PD Linker Page Deduplication**:
   - `f171859219b5` (#47684) rounds DeepSeek-V4.1 dense indexer logits tiles to whole SM waves, eliminating partial tail-wave overhead (**`1.19x`** speedup, `21.42 us -> 18.00 us`).
   - `e232ebf6b0b7` (#47236) enables the Triton softmax top-k router on AMD ROCm for `Qwen3-Next` (`hidden=2048`), achieving a **`1.27x`** router speedup (`15.24 us -> 12.00 us`).
   - `cb0a7a953fa6` (#46395) writes PD decode KV to the linker while skipping KV pages already stored by prefill, cutting redundant linker write traffic by **`68.75%`** (`11/16` pages skipped), and `c757c4bb7c55` (#47732) offsets `kv_events` IPC endpoints per DP rank and finishes shutdown cleanly when the queue is full (`0` collisions/hangs).
3. **TensorRT-LLM GB300 Qwen3.5 AgentX Side-Branch Optimizations & `FORCE_CHUNK` Context Limit Guard**:
   - `5d86481f60bd` (#20815) upstreams the GB300 `Qwen3.5` AgentX side-branch optimizations (**`1.23x`** decode speedup), and `7a62522fb820` (#20900) enforces the `FORCE_CHUNK` context limit when constructing the micro-batch scheduler (`0` context overflows).

---

## 2. Deterministic Offline Verification (Pillar 2 & Pillar 3)

Verified via [`experiments/vllm_kv_dtype_moe_pad_skip_sglang_dsv41_wave_pd_linker_phase179/run.py`](../../experiments/vllm_kv_dtype_moe_pad_skip_sglang_dsv41_wave_pd_linker_phase179/run.py) `--check`:
- `heterogeneous_kv_memory_saving_pct = 43.75`
- `vllm_moe_fused_pad_skip_speedup = 1.21`
- `dsv32_pp_shared_indexer_topk_mismatches = 0`
- `qwen35_397b_b200_nvfp4_tp2_speedup = 1.34`
- `sglang_dsv41_indexer_wave_round_speedup = 1.19`
- `sglang_amd_qwen3_next_triton_router_speedup = 1.27`
- `pd_linker_redundant_write_reduction_pct = 68.75`
- `kv_events_dp_rank_ipc_collisions = 0`
- `trtllm_gb300_qwen35_agentx_speedup = 1.23`

---

## 3. Enterprise Business Scenario Playbook (Pillar 4)

| Dimension | ✅ Good Fit | ⛔ Not For |
| :--- | :--- | :--- |
| **Heterogeneous KV Cache & Long-Context Agentic Serving** | Resolving per-layer `kv_cache_dtype` in vLLM V1 (`43.75%` memory savings, `#56155`) and deploying `Qwen3.5-397B-A17B` B200 NVFP4 TP2 (`#64607`) or GB300 TRT-LLM AgentX (`#20815`). | Forcing uniform BF16 KV caches across hybrid sliding/full layers or bypassing `FORCE_CHUNK` checks in micro-batch creation (`#20900`). |
| **DeepSeek-V3.2 / V4.1 Sparse Attention Serving** | Carrying shared `top-k` indexer tensors across PP stages (`#59961`) and rounding DSv4.1 dense indexer tiles to whole SM waves (`#47684`). | Recomputing or misaligning sparse top-k indices across pipeline boundaries. |
| **PD Disaggregation & Multi-DP Telemetry** | Skipping prefill-stored KV pages during PD decode linker writes (`-68.75%` traffic, `#46395`) and offsetting `kv_events` IPC endpoints per DP rank (`#47732`). | Sharing un-offset IPC endpoints across DP ranks or blocking indefinitely on full telemetry queues at shutdown. |
