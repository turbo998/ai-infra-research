# Phase 128 Deep Dive (Track B): SGLang Declarative Routed KV Stack, Ascend NPU DeepSeek-V4 No-Sync Decode & vLLM BF16 DeepSelect Prefill Indexer / ROCm MXFP8 Norm Fusion / NIXL Push Mode

## 1. Motivation & Core Problem

Production LLM serving engines running hybrid attention (Full + Sliding Window + Mamba state), speculative decoding (EAGLE/EAGLE3/DSPARK), and disaggregated prefill/decode (PD) face three critical bottlenecks:
1. **Combinatorial explosion of unified KV pool variants**: Hand-maintaining separate unified pool classes, PD transfer descriptors, and draft-binding rules per model architecture.
2. **Host-synchronizing ops in single-threaded NPU decode schedulers**: On Ascend NPU running DeepSeek-V4 (DSV4), dynamic boolean indexing (`aclnnNonzeroV2`) and device-to-host (`D2H`) tensor reads stall the host scheduler thread until the in-flight forward drains (`31.4–32.5 ms/round`).
3. **Redundant dtype conversions in sparse-attention prefill indexers and ROCm microscaling**: vLLM upcasted BF16 DeepGEMM indexer logits to FP32 before `deep_select` top-k in prefill, while ROCm MiniMax-M3.1 wrote BF16 outputs after Gemma RMSNorm only to re-read them in a separate `mxfp8_e4m3_quantize` launch.

This report analyzes 9 commits merged on 2026-10-28 across `sgl-project/sglang` (6 commits) and `vllm-project/vllm` (3 commits).

## 2. Key Implementation Paths & Architectural Mechanisms

### 2.1 SGLang 5-Commit Declarative Unified KV Stack & Routed Cache Views (`5433093d83b8`, `298b56359664`, `f80cb3a2fd0f`, `240862923e8d`, `146378629d3e`)
1. **Layer Declarations to Unified Pools (`5433093d83b8`)**: Introduces `sglang/srt/mem_cache/kv/{decl.py,plan.py,build.py}` where each layer declares its byte geometry (`MHAGeometry`/`MLAGeometry`/`StateGeometry`) and lifetime rank (`StateLifetime=1`, `FullLifetime=2`, `WindowLifetime=3`). Layers sharing a lifetime and allocation unit share a sub-pool ordered by lifetime rank under a single `build_unified_pools` entry point.
2. **Storage-Plan Driven PD Transfers (`298b56359664`)**: `get_kv_transfer_buf_infos(pool, storage)` extracts contiguous pointer/length regions directly from `storage.regions(storage.owner)`, registering the owner sub-pool as a single envelope region rather than per-layer entries.
3. **Routed Cache Replacement (`f80cb3a2fd0f`) & Fused Draft Binding (`240862923e8d`)**: Replaces bespoke unified pool subclasses with routed `IdSpaceKind` views (`FULL` vs `SLIDING_WINDOW`) and binds fused speculative drafts (`SpecMode`) to routed page tables across FlashAttention (FA3/FA4) and AITER.
4. **Shared `kv_storage_of` Consumer Interface (`146378629d3e`)**: Unifies PD disaggregation (`prefill.py`, `decode.py`) and beam search (`BeamCoordinator`, `free_member_rows`) onto `kv_storage_of(req_to_token_pool, token_to_kv_pool_allocator)`.

### 2.2 SGLang Ascend NPU DeepSeek-V4 Decode Host-Sync Elimination (`f59cc7e45fd0`)
- **Path**: `python/sglang/srt/hardware_backend/npu/dsv4/{dsv4_allocator.py,dsv4_common_hooks.py}` adds `_compact_c4_loc` and passes `batch.req_pool_indices_cpu` into `maybe_write_dsv4_decode`.
- **Mechanism**: Eliminates all three host-synchronizing ops on the DSV4-NPU decode scheduler chain (`~3.9 ms/round` verify boolean mask `aclnnNonzeroV2`, `32.5 ms/round` alloc boolean mask queue-drain wait, and `31.4 ms/round` `req_pool_indices` D2H sync) by compacting ratio-4 group-closing slots entirely on device via `cumsum -> masked_fill -> scatter` into a host-known upper bound (`min(t, t//4 + bs)` or `t`) and reading the host mirror `req_pool_indices_cpu`, reducing decode host syncs from **`3 -> 0`**.

### 2.3 vLLM DSA Prefill Indexer BF16 DeepSelect Ranking (`c105af429241`)
- **Path**: `vllm/model_executor/layers/sparse_attn_indexer.py`.
- **Mechanism**: When `sparse_indexer_topk_backend="deep_select"`, prefill chunks rank BF16 MQA logits directly via `deep_select_topk(..., end=row_ends)`, dropping `as_fp32_logits` + `top_k_per_row_prefill`. Because BF16 widens to FP32 losslessly, selected top-k values are bitwise identical. On GB200 over 65,536 compressed keys (`top_k=512`), top-k accelerates by **`3.7x–4.9x`** (`66.5 -> 13.6 µs` at 128 rows); on DeepSeek-V4.1-Flash DEP2 128K-prefix prefill, `upcast + top-k` drops from **`39.4 -> 5.7 ms` (`6.912x`)**, total indexer time drops from **`64.5 -> 30.9 ms` (`2.087x`)**, and end-to-end step time drops by **`6.6%`** (`429.8 -> 401.5 ms`).

### 2.4 vLLM NIXL `transfer_mode` Selection & ROCm MiniMax-M3.1 Fused Gemma RMSNorm + MXFP8 Quant (`066d9799347c`, `1cb31be70284`)
- **NixlConnector `transfer_mode` (`066d9799347c`)**: Allows orchestrators that hardcode `"kv_connector": "NixlConnector"` to select `"transfer_mode": "push"` (`NixlPushConnector` using `NIXL WRITE`) or `"pull"` (`NixlPullConnector`) via `kv_connector_extra_config`.
- **ROCm MiniMax-M3.1 Fused Gemma RMSNorm + MXFP8 Quant (`1cb31be70284`)**: Quantizes normed rows per 32-element block in the Gemma RMSNorm epilogue on AMD MI355X (`gfx950`) and passes a `QuantizedActivation` with row-major E8M0 scales directly to `RocmDotScaledMxfp8LinearKernel`. This eliminates separate quant launches in 59 of 60 layers, achieving **`1.41x–2.06x`** speedup on `Norm + Quant`, **`1.07x–1.19x`** on `Norm + Quant + qkv_proj GEMM`, saving **`1.9%`** of total decode GPU kernel time, and improving AgentX P90 interactivity by up to **`1.045x`** at concurrency 16.

## 3. Deterministic Verification Summary

| System / Commit | Dimension | Baseline | New Path | Quantitative Impact |
|---|---|---:|---:|---|
| SGLang KV Stack (`5433093d83b8`..`146378629d3e`) | Unified pool build & PD registration | Variant subclasses / per-layer | `build_unified_pools` / 1 owner region | Unified `SpecMode` & `kv_storage_of` across PD/Beam |
| SGLang NPU DSV4 (`f59cc7e45fd0`) | Decode scheduler Host-Sync stalls | 3 syncs (`32.5 / 31.4 / 3.9 ms`) | `cumsum->masked_fill->scatter` + CPU mirror | **`0` syncs**, eliminates `32.5 ms` queue-drain stall |
| vLLM DSA DeepSelect (`c105af429241`) | GB200 prefill indexer Top-K / E2E | `39.4 ms` (FP32 upcast+topk) / `429.8 ms` | `5.7 ms` (BF16 DeepSelect) / `401.5 ms` | **`4.9x`** Top-K, **`6.912x`** upcast+topk, **`-6.6%`** E2E |
| vLLM ROCm MXFP8 (`1cb31be70284`) | MI355X MiniMax-M3.1 Norm+Quant | Separate Norm + `mxfp8_e4m3_quantize` | Fused Norm epilogue -> `QuantizedActivation` | **`1.41x–2.06x`** Norm+Quant, `-59/60` launches |
| vLLM NIXL (`066d9799347c`) | Connector push-mode selection | Hardcoded class name required | `transfer_mode: "push"` -> `NixlPushConnector` | Zero-code switch to `NIXL WRITE` push disagg |
