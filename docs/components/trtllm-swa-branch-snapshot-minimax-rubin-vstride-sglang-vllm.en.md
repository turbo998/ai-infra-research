# Deep Dive: TensorRT-LLM SWA Prefix-Fork Branch Snapshots, MiniMax-M3 on Rubin (SM107) P128 Hybrid Routing, Separate Context FMHA V Stride, SGLang LTX-2.3 SP RoPE / FLUX.3 Fused SwiGLU-FP8, and vLLM Speculative Decoding Guards

> **Research Question ID**: `q-trtllm-swa-branch-snapshot-minimax-rubin-vstride-sglang-vllm` (Priority 69, Phase 60 Track B)  
> **Upstream Commits Covered**:
> - `NVIDIA/TensorRT-LLM@ae1aa43964e12afb5157ca5d551993a89db2224e` (`#19877`, 2026-10-23)
> - `NVIDIA/TensorRT-LLM@854ccee8a9f0dded7235a0642032de105b3b47f7` (`#20167`, 2026-10-23)
> - `NVIDIA/TensorRT-LLM@d7ec5162ac8d935b21bb5eac719e5eac99664cec` (`#20242`, 2026-10-23)
> - `NVIDIA/TensorRT-LLM@0f3e1d093dceb454f01a02a578cde1aed99d0495` (`#20125`, 2026-10-23)
> - `sgl-project/sglang@2fbd094a8cb641deb099b5b2ac0d362b7eaa6aec` (`#45995`, 2026-10-23)
> - `sgl-project/sglang@c80285ca9adc23946447128926c22e8e7dfb2b80` (`#43809`, 2026-10-22)
> - `sgl-project/sglang@698fa949ecd3f5662a87edcb4aefcb44884a2776` (`#42758`, 2026-10-22)
> - `vllm-project/vllm@37cf712552509d5bf2e67128b80ab435079434fc` (`#63251`, 2026-10-23)
> - `vllm-project/vllm@f3871c3cbf8a608f7ee833ec4c2c5277f77bee41` (`#63339`, 2026-10-23)
> - `vllm-project/vllm@404c2b510e1d73df151d78e5576e1ee7ca0bcaa3` (`#62171`, 2026-10-23)  
> **Companion Deterministic Experiment**: `experiments/trtllm-swa-branch-snapshot-minimax-rubin-vstride-sglang-vllm/run.py`

---

## 1. Architectural Motivation & Problem Statement

In production inference stacks combining Sliding Window Attention (SWA), Rubin (`SM107`) model enablement, multimodal diffusion (LTX-2.3 / FLUX.3), and GDN/Eagle3 speculative decoding, five critical system bottlenecks emerged:

1. **SWA Prefix Cache Loss at Multi-Turn Branch Forks**: In TensorRT-LLM's `KVCacheManagerV2`, sliding-window layers retain only the trailing window at the end of a request. When Request 1 (4096 tokens) diverges from an earlier conversation tree at token 1500, completing Request 1 in a single chunk slides its SWA window to the end of the prompt (`reused = 0` at token 1500). A subsequent sibling Request 2 sharing those first 1472 tokens cannot reuse the SWA state at the fork and must recompute all 1472 tokens from scratch.
2. **MiniMax-M3 Hybrid KV Precision & P128 Routing on Rubin (`SM107`)**: MiniMax-M3 combines dense attention layers, sparse MSA layers, and a shared Eagle3 draft layer using 128-token pages. On Rubin `SM107`, `FlashInferTrtllmGenFmha` supports only 128-token pages (not 64-token dense pages), while uninitialized KV pool tail slots corrupt CSR sparse prefill V dequantization.
3. **Hardcoded Packed V Stride in Context FMHA**: When a model (`qwen4_exp`) slices `v` out of a wide fused projection `[q | k | v | gate | index | pad]` (55 heads of 256 elements, 28,160 B row stride) and runs non-FP8 context MLA, `fmhaRunner.cpp` previously assumed $V$ was packed immediately after $K_{\text{nope}}$ with stride `numKvHeads * (headSizeQkNope + headSizeV)`, reading corrupt columns.
4. **SGLang LTX-2.3 SP RoPE Zero-Offset Bug & ROCm TP>2 Non-Associative All-Reduce**: When LTX-2.3 Batched Classifier-Free Guidance (BCG) recomputed missing RoPE coordinates, it omitted `video_sp_start`, resetting every Sequence Parallel rank to frame 0; meanwhile, on AMD ROCm with `tp_size > 2` and `--disable-custom-all-reduce`, falling back to RCCL violated batch-invariant determinism.
5. **vLLM `cooperative_topk` Sizing by Buffer `stride` Instead of Live Row Length `sl`**: Sizing `per_block = (stride + CS - 1) / CS` using a wide buffer stride (`1,048,576`) rather than live row length `sl` (`4096`) pushed `per_block = 65536 > kFusedMax (512)`, forcing short rows into a slow two-pass global reduction.

---

## 2. Core Implementation Mechanics & Source-Level Analysis

```
+--------------------------------------------------------------------------------------------------+
| Phase 60 Track B: Serving Runtime, KV Cache & Speculative Decoding Architecture                  |
+--------------------------------------------------------------------------------------------------+
| 1. TensorRT-LLM SWA Branch Snapshot (#19877)                                                     |
|    - Fork point = (divergence // tokens_per_block) * tokens_per_block (when reused < point < L)  |
|    - Splits prefill with FORCE_CHUNK at fork point + commits SWA window -> 71.88% sibling reuse  |
|                                                                                                  |
| 2. TensorRT-LLM MiniMax-M3 on Rubin SM107 (#20167) & Separate V Stride (#20242)                  |
|    - Sparse MSA layers keep NVFP4 KV on SM100/103/107; dense + Eagle3 draft stay FP8             |
|    - EXTRA_PAGE_SIZES_ONLY_SMS={107} routes 128-token draft pages to FlashInferTrtllmGenFmha     |
|    - Explicit v_stride_in_bytes (16B TMA aligned) prevents corrupt column reads on sliced V      |
|                                                                                                  |
| 3. SGLang Diffusion & ROCm Determinism (#45995, #43809, #42758)                                  |
|    - LTX-2.3 BCG RoPE passes start_frame = video_sp_start; ROCm refuses RCCL AR when TP > 2     |
|    - FLUX.3 fused_packed_swiglu_fp8_rowwise cuts HBM traffic from 9 B/elem to 5 B/elem (-44.44%) |
|                                                                                                  |
| 4. vLLM Spec-Decode & Top-K Guards (#63251, #63339, #62171)                                      |
|    - cooperative_topk sizes per_block = ceil(sl / CS) instead of stride -> 256x less staging     |
|    - CPU GDN conv-state matches GPU seq_lens[0] == state_len - (width - 2); draft RoPE cap       |
+--------------------------------------------------------------------------------------------------+
```

### 2.1 TensorRT-LLM SWA Prefix-Fork Branch Snapshots (`TensorRT-LLM@ae1aa43964`)

In `tensorrt_llm/_torch/pyexecutor/kv_cache_manager_v2/_core/_kv_cache.py`, when `enable_branch_snapshot=True`, `_swa_branch_snapshot_point` inspects the radix-tree divergence offset `divergence = len(matched)`:
$$\text{point} = \left\lfloor \frac{\text{divergence}}{\text{tokens\_per\_block}} \right\rfloor \times \text{tokens\_per\_block}$$
Whenever $0 < \text{reused} < \text{point} < \text{prompt\_len}$ and $\text{divergence} < \text{num\_lookup\_tokens}$, `LlmRequest.split_chunk_at(point)` splits the prefill at `point` with `ChunkingPolicy.FORCE_CHUNK`, and `RequestScheduler._first_chunk_waits_for_budget` defers the request if the remaining token budget cannot fit the forced first chunk. Once that first chunk executes, `KVCacheManagerV2` commits the SWA window at `point`, enabling any sibling request branching from the same prefix to reuse all `1472` shared tokens directly.

### 2.2 MiniMax-M3 on Rubin (`SM107`) (`TensorRT-LLM@854ccee8a9`) & Separate Context FMHA V Stride (`TensorRT-LLM@d7ec5162ac`)

1. **MiniMax-M3 on Rubin (`SM107`)**:
   - **Per-Layer KV Precision Exemption**: `_exempt_minimax_m3_dense_layers_from_nvfp4_kv` keeps `NVFP4` KV cache on sparse MSA layers (supported on `(10, 0), (10, 3), (10, 7)`) while pinning dense target layers and the shared Eagle3 draft layer to `FP8`.
   - **Page-Size-Aware Backend Selection**: Setting `EXTRA_PAGE_SIZES_ONLY_SMS = {107}` routes 64-token dense target layers to `TrtllmAttention` (SDPA) while routing the 128-token Eagle3 draft layer to `FlashInferTrtllmGenFmha` (`P128`).
   - **Zero-Initialized KV Pools**: `_zero_kv_cache_manager_v2_pools` zeros both primary and draft KV pools at allocation time so CSR sparse prefill V dequantization never reads uninitialized tail bytes.
2. **Separate Context FMHA V Stride (`#20242`)**:
   - `fmhaRunner.cpp` and `quantizeCopyInputToFp8Kernel` accept an explicit `v_stride_in_bytes` (validated for 16-byte TMA alignment in Python), passing `v_token_stride_elements = 14080` (`28,160 B`) when `v` is sliced from `qwen4_exp`'s 55-head fused projection.

### 2.3 SGLang LTX-2.3 SP RoPE Offset, ROCm Determinism Guard, and FLUX.3 Fused SwiGLU-FP8 (`sglang@2fbd094a8c`, `c80285ca9a`, `698fa949ec`)

- **LTX-2.3 BCG Video RoPE SP Shard Start (`#45995`)**: Passes `video_start_frame = video_sp_start or 0` into `_prepare_ltx2_rope_coords_for_bcg`, ensuring 4 SP ranks (16 frames/rank) receive continuous start frames `[0, 16, 32, 48]` instead of `[0, 0, 0, 0]`.
- **ROCm Deterministic All-Reduce Guard Above TP2 (`#43809`)**: `_deterministic_rocm_rccl_check` raises a `ValueError` when `is_hip and enable_deterministic_inference and tp_size > 2 and disable_custom_all_reduce`, preventing fallback to non-batch-invariant RCCL reductions.
- **FLUX.3 Fused Packed SwiGLU + Rowwise FP8 Quantization (`#42758`)**: `_swiglu_fp8_rowwise_kernel` loads `gate` and `up` from packed BF16 `[M, 2*K]`, reproduces both BF16 rounding steps via `round_bf16_to_fp32`, and writes `float8_e4m3fn` directly without materializing a BF16 intermediate, cutting HBM traffic from `9 B/elem` to `5 B/elem` (**-44.44%**).

### 2.4 vLLM `cooperative_topk` Row-Length Sizing, CPU GDN Conv-State, and Draft RoPE Cap (`vllm@404c2b510e`, `f3871c3cbf`, `37cf712552`)

- **`cooperative_topk` Sizing by Live Row Length `sl` (`#62171`)**: Replaces `(stride + CS - 1) / CS` with `(sl + CS - 1) / CS`. For `stride = 1,048,576`, `sl = 4096`, `CS = 16`, `per_block` drops from `65536` to `256 <= kFusedMax (512)`, restoring single-pass execution (**256x** reduction in per-CTA staging span).
- **CPU GDN Spec-Decode `conv_state` Layout (`#63339`)**: Matches GPU `causal_conv1d_update` by checking `seq_lens[0] == state_len - (width - 2)` for full-length speculative rows.
- **V2 Model Runner Draft RoPE Cache Bound (`#63251`)**: `_limit_max_model_len_to_draft` clamps target `max_model_len` to the draft model's `max_position_embeddings` before allocating runner buffers.

---

## 3. Quantitative Evaluation & Benchmark Comparison

Verified via `experiments/trtllm-swa-branch-snapshot-minimax-rubin-vstride-sglang-vllm/run.py`:

| Subsystem / Optimization | Pre-Fix Behavior / Cost | Post-Fix Behavior / Cost | Quantitative Gain / Key Invariant |
| :--- | :--- | :--- | :--- |
| **TRT-LLM SWA Branch Snapshot (`#19877`)** | Sibling recomputes all `2048` tokens (`reused=0`) | Commits SWA window at fork block `1472`; computes `576` tokens | **71.88%** sibling prefill compute saved |
| **TRT-LLM MiniMax-M3 Rubin (`#20167`)** | Missing P64 cubin on SM107 / uninitialized tail NaN | Sparse NVFP4 + Dense/Draft FP8 + P128 draft routing | 100% zero-initialized KV pools; zero tail NaN |
| **TRT-LLM Separate V Stride (`#20242`)** | Assumed packed `k_nope+v` stride on sliced V | Explicit `28,160 B` (`14,080` elems, 16B TMA aligned) | Eliminates corrupt column reads on `qwen4_exp` |
| **SGLang FLUX.3 SwiGLU-FP8 (`#42758`)** | Materializes BF16 activation before FP8 (`9 B/elem`) | Bit-exact double BF16 rounding direct to E4M3 (`5 B/elem`) | **44.44%** HBM traffic reduction |
| **vLLM `cooperative_topk` (`#62171`)** | Sized `per_block=65536` by `stride=1M` (two-pass) | Sized `per_block=256` by live `sl=4096` (single-pass) | Restores single-pass Top-K (**256x** smaller span) |

---

## 4. Architect Design Takeaways & Falsifiable Questions

1. **SWA Branch Snapshots for Agentic Tree Search**: Any serving stack running hybrid SWA models under multi-branch agent workflows or parallel tool exploration should enable `enable_branch_snapshot=True` so prefix-fork windows are committed before sliding away.
2. **Explicit Stride Contracts on Sliced Fused Projections**: When slicing $K$ and $V$ views out of wide fused projections, never assume $V$ shares $K$'s row stride; pass explicit `v_stride_in_bytes` and enforce 16-byte TMA alignment.
3. **Falsifiable Follow-Up Question**: Under high-concurrency multi-fork trees, what is the optimal minimum fork depth (in blocks) before inserting a `FORCE_CHUNK` split so that SWA snapshot commit overhead does not regress single-stream TTFT?
