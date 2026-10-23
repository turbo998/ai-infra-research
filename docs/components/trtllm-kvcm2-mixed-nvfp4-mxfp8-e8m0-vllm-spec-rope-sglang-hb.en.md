# Deep Dive: TensorRT-LLM KVCacheManagerV2 Mixed-Precision NVFP4 KV Cache, MXFP8 E8M0 Subnormal NaN Fix, vLLM V2 Speculative RoPE Clamping, and SGLang Disaggregated Heartbeat Recovery

- **Research Date**: 2026-10-24 (Phase 58 · Priority 67 · Track B)
- **Upstream Evidence Chain**:
  - `NVIDIA/TensorRT-LLM@b4bfec4baf` (`#20452`, *feat(kv_cache): mixed-precision KV cache in KVCacheManagerV2*)
  - `NVIDIA/TensorRT-LLM@348347ce3e` (`#20075`, *[Bugfix] Fix MXFP8 quantization NaN on subnormal E8M0 scale and support Ray named process groups*)
  - `NVIDIA/TensorRT-LLM@96de4749c5` (`#20688`, *perf(speculative): compile _build_replay_work_items_torch*)
  - `NVIDIA/TensorRT-LLM@0f3e1d093d` (`#20276`, *feat(qwen): Qwen3.8-Flash-Next modeling_v2 & NVFP4 KV threshold*)
  - `vllm-project/vllm@37cf712552` (`#63251`, *[Bugfix][V2][SpecDecode] Keep draft model within its RoPE cache on MRV2*)
  - `vllm-project/vllm@44302539f8` (`#55743`, *[Bugfix] Preserve 64-bit integer precision in coerce_to_schema_type*)
  - `vllm-project/vllm@ccd4e4030f` (`#63318`, *feat(frontend): add --return-tokens-as-token-ids to Rust CLI*)
  - `sgl-project/sglang@16aa816060` (`#45922`, *[PD] Reset heartbeat failure count when prefill node fails*)
  - `sgl-project/sglang@30dc2f9430` (`#45982`, *feat(bench): add --input-len-list to bench_one_batch_server*)
- **Evidence Tier**: `implementation` + `own-offline` (`experiments/trtllm-kvcm2-mixed-nvfp4-mxfp8-e8m0-vllm-spec-rope-sglang-hb/run.py`)

---

## 1. Architectural Motivation & Core Problems

Phase 58 Track B addresses four high-impact serving reliability and memory efficiency challenges across TensorRT-LLM, vLLM, and SGLang:

1. **Preserving Attention Sink & Local Window Fidelity Under NVFP4 KV Compression (`TensorRT-LLM #20452`)**: Compressing an entire long-context KV cache into NVFP4 degrades accuracy on initial attention sinks (`first_n` tokens) and recent local context (`last_n` tokens). `KVCacheManagerV2` needed page-granular mixed-precision retention without invasive modifications to every FlashInfer and TRTLLM FMHA kernel.
2. **Silent `0 * inf = NaN` Corruption on Subnormal E8M0 Scales in MXFP8 Quantization (`TensorRT-LLM #20075`)**: Whenever an MoE gated activation (e.g., SwiGLU) produces a near-zero block with `amax in (0, 448 * 2^-127]`, its E8M0 scale is `2^-127` (`fp8SFVal = 0`), which is **subnormal** in IEEE-754 FP32 (`< 2^-126`). Calling `rcp.approx.ftz.f32` flushes `2^-127` to `0.0` and returns `+inf`, turning every zero element in the block into `0.0 * (+inf) = NaN`.
3. **vLLM Model Runner V2 (MRV2) Out-of-Bounds Draft RoPE Gathers (`vllm #63251`) & 64-Bit Integer Tool Coercion Loss (`vllm #55743`)**: Unlike V1, MRV2 cannot dynamically skip speculative decoding when sequence length exceeds the draft model's `max_position_embeddings`, causing OOB indexing into `cos_sin_cache`. Meanwhile, `coerce_to_schema_type(..., "number")` previously called `float(value)` before `int(value)`, corrupting 64-bit integer IDs above `2^53`.
4. **Immediate Re-Eviction of Reconnected Prefill Nodes in SGLang Disaggregated Serving (`sglang #45922`)**: `CommonKVManager._handle_node_failure` did not clear `self.heartbeat_failures[failed_bootstrap_addr]`, leaving a stale count of `3` so a reconnected prefill node was evicted on its very first missed heartbeat.

---

## 2. Mechanism Deep Dive & Code Walkthrough

### 2.1 TensorRT-LLM `KVCacheManagerV2` First-N/Last-N Mixed-Precision NVFP4 KV Cache (`#20452`)

`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py` and `kernels/blackwell/mixed_precision_kv.py` implement `dtype: mixed_nvfp4_bfloat16 | mixed_nvfp4_fp8 | mixed_nvfp4_float16`:

- **`_NVFP4Codec.row_bytes(head_dim)` Formula**:
  $$\text{row\_bytes}(d) = \left\lfloor \frac{\frac{d}{16} \times 9 + 7}{8} \right\rfloor \times 8$$
  For `head_dim = 128`, each head row takes `72 B` (`64 B` E2M1 + `8 B` FP8 scale) vs `256 B` in BF16 (**`3.5556x`** compression).
- **Dual-Pool Page Management & `MixedPrecisionKVStaging`**:
  - Each full-attention layer owns two pools: a primary NVFP4 pool for middle history pages and a `HIGH_PRECISION` pool appended as an extra layer group (`pool_ratio: [0.75, 0.25]`) for the first `mixed_precision_first_n` (e.g., `256`) and last `mixed_precision_last_n` (e.g., `256`) tokens at page granularity.
  - Inside `with attn_metadata.mixed_precision_kv_staging(layer_idx):`, CuTe DSL kernels gather and dequantize NVFP4 history pages alongside native high-precision pages (`_copy_native` via `cute.autovec_copy`) into a temporary staging buffer, cutting per-layer KV footprint at `seq_len = 32,768` (`page_size=32, first_n=256, last_n=256`) by **`70.75%`** (`8,192 KiB -> 2,396 KiB`).

### 2.2 TensorRT-LLM `cvt_warp_fp16_to_mxfp8` Subnormal E8M0 `exp2f_rcp` Fix (`#20075`)

In `cpp/tensorrt_llm/kernels/quantization.cuh`, replacing `reciprocal_approximate_ftz(static_cast<float>(tmpSFVal))` with:
```cpp
float outputScale = vecMax != 0.f ? exp2f_rcp(fp8SFVal) : 0.0f;
```
computes the reciprocal directly from the biased E8M0 exponent (`2^(127 - fp8SFVal) = 2^+127` when `fp8SFVal == 0`), avoiding FP32 subnormal flush-to-zero and eliminating `0 * inf = NaN` across all blocks with `amax in (0, 448 * 2^-127]`.

### 2.3 vLLM V2 Speculative RoPE Clamping (`#63251`), 64-Bit Integer Coercion (`#55743`), and SGLang Heartbeat Reset (`#45922`)

- **vLLM Draft RoPE Cache Alignment (`#63251`)**: For `_CACHE_ONLY_ROPE_TYPES = {"default", "linear", "llama3", "ntk", "proportional"}`, `SpeculativeConfig` raises the draft model's `max_position_embeddings` to match the target `max_model_len`; for context-dependent RoPE (`yarn`, `deepseek`, `dynamic`, `longrope`), `_limit_max_model_len_to_draft()` lowers the V2 engine `max_model_len` to `draft_max_model_len`.
- **vLLM 64-Bit Integer Schema Coercion (`#55743`)**: `coerce_to_schema_type` now tries `int(value)` before `float(value)` for `"number"` fields, preserving exact 64-bit integers such as `"9007199254740993"` (`2^53 + 1`).
- **SGLang Heartbeat Counter Reset (`#45922`)**: `_handle_node_failure` calls `self.heartbeat_failures.pop(failed_bootstrap_addr, None)` so reconnected prefill servers start with `0` accumulated failures.

---

## 3. Architect Exercises & Falsifiable Questions

1. **Quantitative Exercise**: Verify that at `seq_len = 16,384` (`512` pages of `32` tokens) with `first_n = 256` (`8` pages) and `last_n = 256` (`8` pages), `mixed_nvfp4_bfloat16` retains `16` BF16 pages and `496` NVFP4 pages, reducing memory by **`69.64%`**.
2. **Falsifiable Hypothesis**: Passing `"9007199254740993"` through `float()` in Python 3.12 yields `9007199254740992.0` due to 53-bit mantissa rounding, whereas trying `int()` first preserves the exact value.
