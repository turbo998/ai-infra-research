# Serving Engine PD KV Adler-32 Checksum, QSA Prefill Gather, ROCm 32-Bit Triton Ptr Range, DBO Microbatches & MiniMax-M3 VL Deep Dive (Phase 85)

> **Associated Experiment**: `experiments/serving_pd_checksum_qsa_rocm_dbo_minimax_phase85/`  
> **Associated Question**: `q-serving-pd-checksum-qsa-rocm-dbo-minimax-phase85` (`priority: 94`)  
> **中文版本**: [`serving-pd-checksum-qsa-rocm-dbo-minimax-phase85.md`](./serving-pd-checksum-qsa-rocm-dbo-minimax-phase85.md)

---

## 1. Overview & Problem Domain

In Phase 85 (`priority: 94`), we audit 8 production-critical correctness and performance commits across SGLang, vLLM, and TensorRT-LLM covering disaggregated Prefill-Decode (PD) KV verification, sparse attention prefill gather, AMD ROCm 32-bit buffer addressing, Dual-Batch Overlap (DBO), chunked-local KV offloading, and multimodal/LoRA serving:

1. **SGLang Disaggregated KV Adler-32 Checksum & Aux-Buffer Alignment Guard (PR #39729, `5d11819f`)**.
2. **SGLang Vectorized LTX-2 Diffusion Q/K RMSNorm + Split-RoPE Kernel (PR #46111, `a887ac8f`)**.
3. **SGLang Single-Launch QSA Prefill Compressed-Key Gather (PR #45752, `dba22e9b`)**.
4. **SGLang EAGLE Bigram Boundary Token in PD HiCache Decode Restores (PR #45076, `0ff67ee7`)**.
5. **vLLM ROCm 32-Bit Buffer Addressing for `TRITON_ATTN` KV Cache Views (PR #60084, `a55937f2`)**.
6. **vLLM MRV2 DBO Routed-Expert Token Offset & First-Failure Reporting (PR #63372 `2c19f679` & PR #63476 `66bf9315`)**.
7. **vLLM `SimpleCPUOffloadConnector` Chunked-Local Attention Alignment Guard (PR #63266, `65bc6983`)**.
8. **TensorRT-LLM MiniMax-M3 VL Serve-Rendered Prompts & LoRA `ckpt_source` Reload Preservation (PR #20626 `a0cee56d` & PR #20774 `9666d742`)**.

---

## 2. Source-Level Mechanics, Formulas & Quantitative Analysis

### 2.1 SGLang PD KV Adler-32 Checksum & Positional Aux-Buffer Guard (`5d11819f`)
- **Radix Prefix Hit vs Whole-Prompt Digest**: When `--disaggregation-enable-kv-checksum` is combined with `--disaggregation-decode-enable-radix-cache`, a decode-side prefix hit causes prefill to send only the suffix pages while the digest covers the full prompt, turning every prefix hit into a false corruption abort. SGLang detects this combination and safely disables the digest check, while admitting cross-layout peers (differing TP width, PP stage, or page size) with a single warning.
- **Aux-Buffer Bounds**: Both Mooncake (`mooncake/conn.py`) and MORI (`mori/conn.py`) pair aux metadata buffers positionally and clamp transfers to $N_{\text{xfer}} = \min(|\text{prefill\_aux\_ptrs}|, |\text{dst\_aux\_ptrs}|)$ with an explicit `check_peer_aux_buffer_count` diagnostic.

### 2.2 SGLang Vectorized LTX-2 Split-RoPE (`a887ac8f`) & QSA Prefill Gather (`dba22e9b`)
- **LTX-2 16-Byte Vectorization (`a887ac8f`)**: `ltx2_qknorm_split_rope_kernel<kRoundIntermediates, kVec=true>` processes `kWidth = 8` consecutive complex pairs (`16` BF16 elements = `32 B`) per thread per step using 16-byte loads while preserving exact scalar FMA rounding (`y0 = fmaf(-sin, x1, cos*x0)`, `y1 = fmaf(sin, x0, cos*x1)`), cutting global memory transactions by **`8x`**.
- **Single-Launch QSA Prefill Gather (`dba22e9b`)**: Exploiting page-aligned QSA allocation where $\text{compressed\_slot}(s, b) = \lfloor \text{token\_slot\_table}[s, b \times \text{RATIO}] / \text{RATIO} \rfloor$, `_gather_qsa_compressed_keys_kernel` packs all sequences' complete compressed keys in **`1` Triton launch** (down from $B+1$ launches) with **0 host syncs** when `prefill_sequence_lengths_cpu` is available.

### 2.3 SGLang EAGLE HiCache Bigram Boundary Token (`0ff67ee7`)
Because an EAGLE tree keys $N$ bigrams with $N + 1$ token IDs, `_try_hicache_queue_load_back` computes:
$$\text{rematch\_token\_len} = \text{pm.decode\_prefix\_len} + \mathbb{I}(\text{tree\_cache.tree\_core.is\_eagle})$$
ensuring the final bigram is included when re-matching restored device prefixes.

### 2.4 vLLM ROCm 32-Bit Buffer Addressing for `TRITON_ATTN` (`a55937f2`)
Triton's AMD backend emits 32-bit buffer descriptors (`buffer_load_dword`) only when pointer offsets are proven $< 2\text{ GiB}$ (`2,147,483,648 B`), checking `tensor.ptr_range()` before falling back to total storage size. For a 32-layer `8 GiB` unified KV cache, per-layer `256 MiB` views previously fell back to 64-bit addressing. `set_triton_ptr_range(kv_cache, key_cache, value_cache)` binds `view.ptr_range = functools.partial(int, region_end - view.data_ptr())`, restoring 32-bit buffer instructions for `key_cache` (`256 MiB`) and `value_cache` (`128 MiB`).

### 2.5 vLLM MRV2 DBO Fixes (`2c19f679`, `66bf9315`) & Chunked-Local CPU Offload (`65bc6983`)
- **DBO Routed-Expert Offset (`2c19f679`)**: `RoutedExpertsCapturer.capture()` slices `self.device_buffer[offset : offset + len(local_topk_ids), layer_id]` using `dbo_current_ubatch_token_offset()`, preserving all `128` rows across two 64-token microbatches instead of overwriting `[0:64]`.
- **DBO First-Failed Microbatch (`66bf9315`)**: Replaces `min(errors)` with `next(iter(errors))` so a root-cause failure in microbatch 1 is reported instead of microbatch 0's subsequent `dbo_register_recv_hook` error.
- **Chunked-Local Alignment Guard (`65bc6983`)**: In `SimpleCPUOffloadConnector.get_num_new_matched_tokens`, rejects CPU suffix reuse whenever `num_computed_tokens % spec.attention_chunk_size != 0` (e.g., `16 % 64 != 0`), preventing suffix-relative chunk boundaries from diverging from absolute token positions.

### 2.6 TensorRT-LLM MiniMax-M3 VL & LoRA `ckpt_source` (`a0cee56d`, `9666d742`)
- **MiniMax-M3 VL (`a0cee56d`)**: Uses bare template markers `]<]image[>[` and `]<]video[>[`, adds `TextPrompt.chat_template_applied = True` to prevent double chat-template rendering on `trtllm-serve` prompts, and skips vision tower execution when all placeholder tokens are cached or outside the current chunk.
- **LoRA `ckpt_source` (`9666d742`)**: Propagates `py_lora_ckpt_source` on `LlmRequest` so `PeftCacheManager.add_request_peft` reloads evicted adapters using the request's original checkpoint format (`"nemo"` vs `"hf"`).

---

## 3. Architect Design Exercises & Falsifiable Questions

1. **Calculation Exercise (ROCm `ptr_range`)**: For a 40-layer unified KV allocation of `192 MiB` per layer (`7.5 GiB` total), verify that `set_triton_ptr_range` bounds each layer's `key_cache` to `192 MiB` and `value_cache` to `96 MiB`, both strictly below the `2 GiB` (`2,147,483,648 B`) CDNA 32-bit buffer limit.
2. **Falsifiable Hypothesis**: If `SimpleCPUOffloadConnector` admits a 48-token CPU suffix after a 16-token GPU prefix on a `64`-token chunked-local attention layer, tokens at absolute positions `64..79` are evaluated with mismatched chunk boundaries relative to a cold prefill.
