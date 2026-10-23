# Deep Dive: TensorRT-LLM KDA Fused Prefill Short Conv & GVR v2 Top-K, SGLang GLM-5.3 Flash Padded CP BCG & MegaGate Mask Reuse, and vLLM ROCm gfx942 32-Bit Offset Fix

> **Language / 语言**: [English](sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942.en.md) | [中文](sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942.md)
> **Question ID**: `sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942` (`priority: 76`, Phase 67 Track B)
> **Evidence Class**: `implementation` + `own-offline`

---

## 1. Problem Statement & Architectural Context

Across hybrid linear attention (KDA / Gated DeltaNet), multimodal MoE gating, and AMD Instinct (`gfx942` / `gfx950`) decode kernels, three production bottlenecks emerged:

1. **Catastrophic Cancellation in TensorRT-LLM KDA Fast SiLU & GVR Top-K Dispatch**:
   - In the CuTe-DSL KDA prefill fused short convolution (`fused_prefill_conv.py`), computing fast SiLU via $\text{silu}(x) = h + h \cdot \tanh(h)$ ($h = x/2$) using `tanh.approx.f32` keeps relative error `< 1.3%` for $h \ge -2$, but suffers catastrophic cancellation as $h \to -\infty$ ($\tanh(h) \to -1$, exceeding `3900%` error at $h = -6$).
   - Meanwhile, GVR decode Top-K self-sampling (`gvr_topk_decode_self_sampling.py`) computed shared memory size from `NBSEL` instead of the rung's actual `NBH` and risked 32-bit mask overflow when `det=True` and `VPT > 8`.
2. **SGLang GLM-5.3 Flash Padded CP BCG Token Conflation & DSV4.1 Redundant MegaGate Masks**:
   - Under Context Parallelism (`CP`) + Batch-level CUDA Graph (`BCG`), conflating the captured static graph capacity (`static_num_tokens`) with live global tokens (`total_seq_lens`) and per-rank local tokens (`per_rank_logical_token[cp_rank]`) corrupts KDA sequence reordering, leaks padded tail rows into attention outputs, or truncates KV cache writes.
   - In DeepSeek-V4.1's 60 MoE layers, `MegaGate` rebuilt identical `(valid, image)` row masks (`torch.arange(num_tokens) < bound` and `image_input_ids == image_token_id`) in every layer (`120` small kernel ops per forward pass).
3. **vLLM ROCm `gfx942` (MI300X) Triton Register-Spill Miscompilation & `gfx950` FlyDSL Duplication**:
   - Typing `output_stride_1: tl.int64` in `kernel_paged_attention_2d` triggered an LLVM AMDGPU register-spill miscompilation under high register pressure on `gfx942`, overwriting `output_stride_1` with an unrelated spilled SGPR and leaving output heads unwritten (`NaN`).

---

## 2. Upstream Implementation & Mechanism Analysis

### 2.1 TensorRT-LLM: KDA Fused Prefill Short Conv (`#20653`) & GVR v2 Top-K (`#20106`)

1. **KDA Fused Prefill Short Conv + Warp-Vote Guarded Hybrid SiLU (`222676c8d7bd`)**:
   - `fused_prefill_conv.py` fuses the 4-tap depthwise short convolution, SiLU activation, and indexed recurrent state pool update (`slot == -1` bypass and `-pool <= slot < -1` negative index wrap) into one CuTe-DSL kernel, while deleting `7,214` lines of legacy KDA kernels (`akk_inverse.py`, `fused_k123.py`, `fused_k1234.py`, `k4_persistent.py`) in favor of vendored FlashInfer `kda_chunked_bt16`.
   - **Hybrid SiLU Guard**: Evaluates `h + h * tanh.approx(h)` across each 8-token thread group, performs **1 warp vote** to check whether any thread has `min(h) < -2.0` (`x < -4.0`), and recomputes only flagged groups using subnormal-preserving `x * e / (1 + e)` ($e = \exp(-|x|)$).
2. **GVR v2 Top-K Decode Vectorization & Rung Expansion (`4ccdca1e8492`)**:
   - Vectorizes `scan_cross_w` SMEM histogram copies (`num_bits_per_copy = 32 * BPT`), widens the register crossing-bin gate `QC` from `96` to `160`, adds `short_wide` and `long_register` `NB=1024` rungs, fixes `IMGOFF = NBH` and `smem = (NBH + 2 * CMP) * 4` (saving `4,096 B` SMEM/CTA when `NBH=1024 < NBSEL=2048`), and transforms `(BLK, VPT) -> (BLK * 2, (VPT + 1) // 2)` when `det=True and VPT > 8`.

### 2.2 SGLang: GLM-5.3 Flash Padded CP BCG (`#43774`), DSV4.1 MegaGate Cache (`#46071`), & Granian Keep-Alive (`#45760`)

1. **GLM-5.3 Flash `InterleaveBcgAdapter` Three-Way Token Separation (`ea226cf78267`)**:
   - Separates `metadata.graph_num_tokens = static_num_tokens` (used by captured KDA sequence reordering), `self.moe_local_token_count` (scalar GPU tensor holding `per_rank_logical_token[cp_rank]`, used by `attention_query_tokens` to slice Q and `attention_output_tokens` to zero padded tail rows `[raw:]`), and `cache_tokens` (global `total_seq_lens` for KV cache writes).
2. **DeepSeek-V4.1 `mega_gate_forward_scope` Row-Mask Reuse (`5f7542d7627f`)**:
   - Caches `(valid, image)` row masks on `forward_batch._mega_gate_row_masks` keyed by `(num_tokens, id(bound), id(image_input_ids))` once per forward pass, reducing mask-building tensor ops across 60 MoE layers from `120` to **`2` (`60.0x` reduction, `-98.33%`)**.
3. **Granian HTTP/1.1 Keep-Alive Propagation (`15dc838f9bf5`)**:
   - Maps `SGLANG_TIMEOUT_KEEP_ALIVE` (seconds) to `HTTP1Settings(header_read_timeout=timeout_s * 1000)` milliseconds in `_run_granian_server`.

### 2.3 vLLM: ROCm `gfx942` 32-Bit Head Offset (`#63268`) & `gfx950` FlyDSL Unification (`#61649`)

- **ROCm `gfx942` Triton Paged Attention Fix (`18df4cd9a98b`)**:
  - Advances `output_ptr += cur_batch_in_all_start_index * output_stride_0` in 64-bit pointer arithmetic while keeping `output_offset = query_head_idx * output_stride_1` in 32-bit `int32` (`8,064` max elements for 64 heads × 128 head_dim), avoiding the 64-bit multiply SGPR spill miscompilation on `gfx942`.
- **ROCm `gfx950` FlyDSL TurboQuant/UltraQuant Unification (`29cf023770a0`)**:
  - Migrates `tq_decode.py` to public `buffer_view` and hardware `ds_read_tr16_b64` in-register transpose, unifying `GQA-{6, 8, 16}` (deleting `875` duplicate lines in `tq_decode_gqa6.py`) and folding `base_byte_offset` into the base pointer so 32-bit `voffset` never overflows the `4 GiB` V# descriptor limit.

---

## 3. Deterministic Verification Summary (`own-offline`)

Verified by `python experiments/sglang-glm53-cp-bcg-trtllm-kda-conv-vllm-gfx942/run.py --check`:

| Metric / Invariant | Baseline / Before | Optimized / After | Deterministic Finding |
|---|---|---|---|
| TRT-LLM KDA SiLU rel. error at $h=-6$ | `> 3953%` (cancellation) | Warp-vote exact fallback (`0%` cancel) | Error strictly `< 1.35%` for $h \ge -2$; `7,214` legacy lines deleted |
| TRT-LLM GVR v2 Top-K (`NBH=1024, CMP=2560`) | `28,672 B` SMEM, `QC=96` | `24,576 B` SMEM, `QC=160` | Saves **`4,096 B` SMEM/CTA**; `VPT=10` det remaps to `(1024, 5)` |
| SGLang GLM-5.3 CP=4 (`static=1024, live=770`) | Static/live token conflation | Per-rank logical `[193, 193, 192, 192]` | Padded tail `[63, 63, 64, 64]` zeroed; DSV4.1 masks reduced **`60.0x`** |
| vLLM `gfx942` head offset & `gfx950` V# offset | `int64` SGPR spill `NaN` / `4.91 GiB` overflow | `int32` (`8,064` elems) / `base_byte_offset` | Eliminates `gfx942` `NaN` miscompile; deletes `875`-line `gqa6` clone |
