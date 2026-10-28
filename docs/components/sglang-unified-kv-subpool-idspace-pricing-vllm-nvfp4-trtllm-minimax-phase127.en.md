# SGLang Unified KV Subpool/ID-Space/Capacity Pricing & HiCache Recurrent-State Event Forwarding, vLLM Spec-Decode DP Graph Padding & DeepSeek-V4.1 Tiled NVFP4 KV Gather, and TensorRT-LLM MiniMax-M3 NVFP4 Weight Loader Analysis (Phase 127 · Priority 136)

> **Language / 语言**: [中文 (`sglang-unified-kv-subpool-idspace-pricing-vllm-nvfp4-trtllm-minimax-phase127.md`)](sglang-unified-kv-subpool-idspace-pricing-vllm-nvfp4-trtllm-minimax-phase127.md) | **English (Current)**

## 1. Executive Architectural Findings

In Phase 127 Track B, we audited **9 upstream commits** across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`:

1. **SGLang 6-Commit Unified KV Memory-Cache Refactor Series (`sglang@306a699a`, `3b597cb7`, `cf3d21b0`, `5f6075fa`, `ccc71fa5` & `3355d668`)**:
   - **Single `MultiEndedAllocator` & Centralized Capacity Pricing (`ccc71fa5` & `5f6075fa`)**: Consolidates 3 sub-pool allocator classes into a single `MultiEndedAllocator` and centralizes shared-gap byte pricing for Mamba/recurrent slots, eliminating **`12.5% -> 0.0%` admission over-commit**.
   - **KV Subpool Views & `IdSpaceKind` Address Translation (`3b597cb7`, `cf3d21b0` & `306a699a`)**: Routes layer KV access through subpool views instead of ad-hoc `layers_mapping` dicts, formalizes virtual-to-physical translation via `IdSpaceKind`, and centralizes recurrent state ownership.
   - **HiCache Completion Event Forwarding to Recurrent State (`3355d668`)**: Forwards `set_hicache_transfer_done_event` to the recurrent state allocator alongside full and SWA attention allocators, reducing unready Mamba state reads during HiCache offload/reload to **`0`**.
2. **vLLM Speculative Decode Adaptive Draft Prefill Graph Padding (`vllm@5118d990`) & DeepSeek-V4.1 Tiled NVFP4 Compressed-KV Dequant-Gather (`vllm@6abdff59`)**:
   - Adds `max_query_len` to `DPSyncState` so DP ranks agree on the query-length bound before dispatching varlen decode/draft prefill CUDA graphs (**`0` cross-DP graph padding mismatches**).
   - Tiles `_dequantize_and_gather_k_nvfp4_kernel` by `TILE` tokens per iteration and places each 4-bit `e2m1` nibble into FP16 bits (`sign` at 15, `exp:mantissa` at `11:9` = `e2m1 * 2^-14`) multiplied by `scale * 2^14` in FP32, achieving a **`1.753x` speedup (`48.0 -> 27.38 us`)** at `TILE=16`.
3. **TensorRT-LLM MiniMax-M3 `compressed-tensors` NVFP4 Weight Loader (`TensorRT-LLM@1f9b7072`)**:
   - Extracts `convert_compressed_tensors_nvfp4_weights` into `hf/weight_mapper.py` for both Laguna and MiniMax-M3, inverting `.weight_global_scale` and `.input_global_scale` with safe zero-scale handling for uncalibrated experts (**`0` divide-by-zero NaNs**).

## 2. Deterministic Verification Summary (`experiments/sglang_unified_kv_subpool_idspace_pricing_vllm_nvfp4_trtllm_minimax_phase127`)

| Subsystem / Mechanism | Baseline Behavior | Phase 127 Upgrade | Verified Quantitative Gain |
|---|---|---|---|
| SGLang Unified KV Sub-Pool & Pricing | 3 separate allocators + policy-side Mamba slot math | Single `MultiEndedAllocator` + recurrent HiCache event forwarding | **Overcommit `12.5% -> 0.0%`, `0` unready reads** |
| vLLM Spec-Decode DP Graph Padding (`5118d990`) | Only synced `uniform_token_count` across DP | `DPSyncState.max_query_len` gates varlen decode graphs | **`0` cross-DP CUDA Graph mismatches** |
| vLLM DSv4.1 Tiled NVFP4 KV Gather (`6abdff59`) | 1-token/iter (`TILE=1`) decode (`48.0 us`) | `TILE=16` + bitwise FP16 `2^-14 * 2^14` decode (`27.38 us`) | **`1.753x` speedup** |
| TRT-LLM MiniMax-M3 NVFP4 Loader (`1f9b7072`) | Missing mapper & divide-by-zero on uncalibrated expert | Shared `convert_compressed_tensors_nvfp4_weights` | **`53` duplicate LOC removed, `0` NaNs** |
