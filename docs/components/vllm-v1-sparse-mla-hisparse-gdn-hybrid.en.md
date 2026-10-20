# vLLM V1 FlashMLA Sparse Multi-Generation KV Layouts, NVIDIA `HiSparse` Hot-Cache & `GDN` Hybrid Checkpoint Architecture

> **Language / 语言**: [中文 (Chinese)](vllm-v1-sparse-mla-hisparse-gdn-hybrid.md) | **English**

## 1. Architectural Motivation: Memory & Scheduling Engineering for Sparse MLA and Hybrid Linear Attention

At 128K–1M context lengths, serving Sparse Multi-Head Latent Attention (Sparse MLA across DeepSeek V3.2, V4, V4.1, and Blackwell NVFP4) alongside hybrid GatedDeltaNet (`GDN_ATTN`) recurrent states introduces three system challenges:
1. **Byte-Exact KV Cache Layouts & Coalesced Dequantization**: Compressing MLA KV rows from `1,152 B` (BF16) down to `656 B`, `584 B`, `528 B`, and `352 B` (`nvfp4_ds_mla`) while maintaining TMA stride alignment and single-thread coalesced scale loads;
2. **GPU Hot-Cache Sizing & Shared-Memory Residency Guardrails in `HiSparse`**: When offloading full context pages to a pinned host pool (`HiSparseHostPool`), speculative decoding (`max_query_len = 4`) unions top-$k$ selections across multiple verification queries. Under-sizing the GPU hot buffer breaks residency, while over-sizing violates per-block shared memory or the `int16` slot-index limit (`32,768`);
3. **Cross-Layer Index Conversion & GDN 64-Token Checkpoint Alignment**: Converting logical top-$k$ indices to physical block-table rows across 61 layers or recording CUDA events inside piecewise CUDA graphs causes severe launch overhead or `cudaErrorInvalidValue` crashes.

Grounded in first-hand source code from **`vllm-project/vllm` (`commit 5285b9b8646d6d0e8b869c9f5c6d277ccfce637b`)**—specifically `vllm/v1/attention/backends/mla/flashmla_sparse.py`, `vllm/v1/attention/backends/mla/index_group.py`, `vllm/v1/hisparse/runtime.py`, and `vllm/v1/attention/backends/gdn_attn.py`—this deep dive derives the exact byte layouts and runtime invariants.

---

## 2. First-Hand Source Code Mechanics (`implementation`)

### 2.1 `vllm/v1/attention/backends/mla/flashmla_sparse.py`: Four Quantized Sparse MLA KV Formats & `8x4 -> 4x8` Scale Permutation

In `flashmla_sparse.py` (`blob 9722256f889bcedf5ae2ee9a30ee6c7bb7828a06`), FlashMLA Sparse distinguishes four quantized KV cache formats directly from bytes-per-token:

1. **DeepSeek V3.2 (`656 Bytes/token`)**: `512 B` `float8_e4m3` NoPE + `16 B` (`4 × float32` scales, 1 per 128 elements) + `128 B` (`64 × bfloat16` unquantized RoPE);
2. **DeepSeek V4 Split-Page Layout (`584 Bytes/token`)**: `448 B` `float8_e4m3` NoPE + `128 B` (`64 × bfloat16`) RoPE + `8 B` (`7 × ue8m0 + 1B pad`, 1 per 64 elements). Within each page, all `576 B` data rows are stored first (matching the decode kernel's `576 B` TMA stride), followed by all `8 B` scale rows;
3. **DeepSeek V4.1 MXFP8 Interleaved Layout (`528 Bytes/token`)**: All `512` dims (**including the 64 RoPE dims**) are quantized as `float8_e4m3` (`512 B`), followed immediately per token by `16 B` (`16 × ue8m0` scales, 1 per 32 elements, `528 B` TMA stride);
4. **SM100 `nvfp4_ds_mla` (`352 Bytes/token`, Blackwell only)**:
   - First `256 B`: `512 × e2m1` NoPE packed 2 per byte (low nibble = even element);
   - Next `64 B`: `64 × float8_e4m3` **unscaled RoPE** (`e4m3`'s 4 exponent bits span the RoPE dynamic range unaided);
   - Last `32 B`: `32 × float8_e4m3` NoPE scales (1 per 16 elements), permuted via an **`8×4 -> 4×8` transpose**:
     $$\text{byte\_offset}(s) = 8 \times (s \mathbin{\&} 3) + (s \gg 2), \quad 0 \le s < 32$$
     so the 8 scale blocks required by a single dequantization thread (`s = 0, 4, 8, 12, 16, 20, 24, 28`) reside at contiguous bytes **`[0, 1, 2, 3, 4, 5, 6, 7]`**.

### 2.2 `vllm/v1/hisparse/runtime.py` & `index_group.py`: `HiSparse` Hot-Cache Sizing & Shared-Memory Residency Equations

In `ResolvedHiSparseConfig.from_vllm_config` (`blob 87916ba4ad945f09b24543d02b33fdcf709f3efd`):
- Worst-case union of host rows per request step: $\text{max\_union\_rows} = Q_{\max} \times K_{\text{top}}$;
- Default hot-cache buffer size with 1 top-$k$ of LRU slack: $H_{\text{hot}} = (Q_{\max} + 1) \times K_{\text{top}} \le 32768$ (`int16` slot limit);
- `_check_residency_shared_memory` enforces per-block shared memory for `hisparse_resolve_residency`:
  $$\text{required\_smem\_bytes} = 4 \times \left(2(\text{max\_union\_rows} + 1) + 3\left\lceil \frac{H_{\text{hot}}}{32} \right\rceil + 7\right) + 2 H_{\text{hot}}$$

In `SparseMLAIndexGroup._convert_once` (`blob ba787b38da552acbd58c58ca9b2f6a5f00b85fc6`), layer `0` converts `logical_topk_indices -> physical_topk_indices` once on `side_stream` for all layers in the group, skipping graph-local `cudaEventRecord` when `in_piecewise_cudagraph()` is active.

### 2.3 `vllm/v1/attention/backends/gdn_attn.py`: FlashInfer 64-Token Checkpoint Planner

In `plan_flashinfer_checkpoints` (`blob 660ef5387b1752953adf8dbea8f8f81cfb8f680f`), `every_n_tokens = math.gcd(*offsets)` (a multiple of `FLASHINFER_GDN_CHECKPOINT_ALIGNMENT = 64`), `cu_starts[i+1] = cu_starts[i] + query_len // every_n_tokens`, and `rows[first + i] = cu_starts[i] + offset // every_n_tokens - 1`.

---

## 3. Deterministic Offline Evaluation (`own-offline`)

Verified by [`experiments/vllm_v1_sparse_mla_hisparse_sim/run.py`](../../experiments/vllm_v1_sparse_mla_hisparse_sim/run.py) and [`results.json`](../../experiments/vllm_v1_sparse_mla_hisparse_sim/results.json) (`61` layers, `32 × 128K` tokens):

| Sparse MLA KV Format | Bytes / Token | TMA Stride | `32×128K` (61L) KV Footprint | Compression vs BF16 | Architectural Layout Highlight |
|---|---|---|---|---|---|
| `bf16_mla_baseline` | `1,152 B` | `1,152 B` | `274.50 GiB` | `1.000x` | 512 BF16 NoPE + 64 BF16 RoPE |
| `deepseek_v3_2_fp8_ds_mla` | **`656 B`** | `656 B` | `156.31 GiB` | **`1.756x`** | 512B FP8 NoPE + 16B FP32 Scale + 128B BF16 RoPE |
| `deepseek_v4_fp8_split_page` | **`584 B`** | **`576 B`** | `139.16 GiB` | **`1.973x`** | Split page: `576B` data rows first, `8B` (`7×ue8m0+1B`) scale rows after |
| `deepseek_v4_1_mxfp8` | **`528 B`** | `528 B` | `125.81 GiB` | **`2.182x`** | Quantized RoPE + `16×ue8m0` (`gran=32`) interleaved per token |
| `sm100_nvfp4_ds_mla` | **`352 B`** | `352 B` | **`83.88 GiB`** | **`3.273x`** | 256B E2M1 + 64B unscaled FP8 RoPE + `8*(s&3)+(s>>2)` contiguous scales |

**`HiSparse` & `SparseMLAIndexGroup` Gains (`K_top = 2048, L_seq = 128K`)**:
- Single-token decode (`Q_max = 1`): `H_hot = 4,096` rows, **`26,148 B`** SMEM;
- 4-token MTP speculative decode (`Q_max = 4`): `H_hot = 10,240` rows (**`7.8125%`** of 128K context, **`12.8x`** GPU HBM reduction), **`89,892 B`** SMEM;
- `SparseMLAIndexGroup._convert_once`: Reduces index conversion launches across 61 layers by **`61x`** (`0.5795 -> 0.0095 ms`).

---

## 4. Architect Decision Guidelines & Falsifiable Questions

1. **Format Selection**: Use `sm100_nvfp4_ds_mla` (`352 B/token`) on Blackwell SM100 (`block_size=64`) for **`3.273x`** KV compression, and `528 B` V4.1 MXFP8 or `656 B` V3.2 on Hopper SM90.
2. **Falsifiable Follow-Up**: When `HiSparseHostPool` shares a single local `mmap` across `TP=8` (`use_shared_hisparse_host_pool`), how does `_hisparse_registration_ranges` modular arithmetic (`pow(block_bytes // divisor, -1, period)`) behave when `block_bytes` is `656 * 64 = 41,984 B`?
