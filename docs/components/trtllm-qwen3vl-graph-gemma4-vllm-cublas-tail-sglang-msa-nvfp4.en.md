# TensorRT-LLM Multimodal Encoder CUDA Graph & Gemma4 Budgeting, vLLM cuBLAS Cleanup & Offloading Tails, and SGLang MSA-v1 NVFP4 KV Kernels

> 中文版: [trtllm-qwen3vl-graph-gemma4-vllm-cublas-tail-sglang-msa-nvfp4.md](trtllm-qwen3vl-graph-gemma4-vllm-cublas-tail-sglang-msa-nvfp4.md)

## 1. Context and Core Architectural Problems

Production multimodal and hybrid-attention inference engines (TensorRT-LLM, vLLM, SGLang) face five cross-layer coordination challenges:

1. **Padding Contamination and Zero-Length Context Rejection in Multimodal Vision CUDA Graphs (`TensorRT-LLM #20213`)**: Appending padding tokens to the last real image context in full-attention vision encoders (Qwen3-VL, RADIO, GLM5-Next) lets real tokens attend to padding rows, while an empty dummy context is rejected by the attention backend.
2. **Encoder Activation Under-Budgeting in Gemma4 Multimodal Requests (`TensorRT-LLM #20663`)**: Gemma4 vision tower activations scale with the **padded patch count** rather than the valid pooled soft-token count, causing OOM under multi-image batches if scheduled by output token count alone.
3. **Retained cuBLAS Scratch, EAGLE Partial-Tail Mismatch, and 64-Bit ROCm Pointer Fallback (`vLLM #63451, #61019, #60084`)**: `WeightCacheDaemon` retained ATen cuBLAS workspaces after tearing down NCCL groups; `OffloadingConnector` keyed EAGLE attention tails at the Mamba boundary instead of one core alignment unit later; and ROCm `TRITON_ATTN` lost 32-bit `buffer_load`/`buffer_store` addressing on per-layer views into a shared $>2\text{ GiB}$ KV allocation.
4. **Nibble Drift Between IEEE Division and `rcp.approx.ftz.f32` in SGLang `MSA-v1` NVFP4 KV Quantization (`SGLang #46139, #46140`)**: Replacing `rcp.approx.ftz.f32` with exact IEEE division in 16-element NVFP4 KV quantization shifts ~1 ULP across E2M1 ties and alters `~0.1%` of output nibbles on B300.
5. **False Corruption Alarms When Combining Disaggregated KV Checksums with Decode RadixCache (`SGLang #39729`)**: A decode-side prefix hit transfers only the suffix KV blocks while the prompt digest covers the whole sequence, causing every cache hit to fail checksum verification.

---

## 2. First-Hand Source Code Audit

### 2.1 TensorRT-LLM `EncoderGraphIOAdapter` Isolated Dummy Context and Gemma4 Budgeting (`TensorRT-LLM@7d391a06d7fb`, #20213; `abb94ab8fc1c`, #20663)

- **Isolated 1-Token Dummy Padding Context (#20213)**: `EncoderGraphIOAdapter` and `MultimodalEncoderMixin` unify CUDA graph capture across **Qwen3-VL**, **RADIO (`vision`)**, and **GLM5-Next (`image`)**. When `enable_padding=True`, each configured bucket `(total_tokens, num_contexts)` is captured with:
  $$\text{captured\_total\_tokens} = \text{bucket.total\_tokens} + 1, \quad \text{captured\_num\_contexts} = \text{bucket.num\_contexts} + 1$$
  Reserving one extra token guarantees that the isolated dummy padding context is always non-empty even when real tokens fill `bucket.total_tokens`.
- **Gemma4 Multimodal Encoder Item Budgeting (#20663)**: `modeling_gemma4mm.py` computes per-item encoder cost using padded patches and stride-2 audio subsampling:
  $$\text{padded\_patches\_per\_frame} = \text{max\_soft\_tokens} \times (\text{pooling\_kernel\_size})^2, \quad \text{AUDIO\_SUBSAMPLING\_FACTOR} = 4$$
  For `max_soft_tokens = 256, pooling_kernel_size = 3`, each frame budgets $256 \times 9 = 2304$ padded patches, while 3000 mel audio frames yield $3000 / 4 = 750$ subsampled audio tokens.

### 2.2 vLLM cuBLAS Workspace Cleanup, EAGLE Partial-Tail Proof Margin, and ROCm 32-Bit Range (`vllm@b3466364b69d`, #63451; `fc0cfcdc05b1`, #61019; `a55937f25528`, #60084)

- **`WeightCacheDaemon.release_distributed()` cuBLAS Cleanup (#63451)**: Calls `torch._C._cuda_clearCublasWorkspaces()` before `torch.accelerator.empty_cache()` so cached ATen BLAS workspaces do not stay pinned for the daemon's lifetime.
- **EAGLE/MTP Partial-Tail Proof Margin (#61019)**: When core drops one alignment unit past a fine-grained hit on an EAGLE/MTP hybrid model (Mamba checkpoint at `24` on a 30-token prompt), `scheduler._partial_tail_key_tokens(group, 24)` keys Mamba at `24` and the EAGLE attention group at:
  $$\text{proof\_tokens} = 24 + (\text{cache\_hit\_alignment\_tokens} \;\text{if not None else}\; \text{block\_size}) \in \{28, 40, 40\}$$
- **ROCm `set_triton_ptr_range` for `< 2 GiB` Layer KV Views (#60084)**: Bounds each layer view's `ptr_range` to `region_end - view.data_ptr()` (e.g., `512 MiB` per layer within a `32 GiB` 64-layer allocation), restoring Triton's 32-bit `buffer_load`/`buffer_store` instructions on AMD GPUs.

### 2.3 SGLang `MSA-v1` NVFP4 KV Kernels, MiniMax-M3.1 NVFP4, and Disaggregated KV Checksum (`sglang@174bd2976d34`, #46139; `a99c7d050fae`, #46140; `5d11819f3a7a`, #39729)

- **`MSA-v1` NVFP4 KV Quantization & `rcp.approx.ftz.f32` (#46139)**: Quantizes 16-element BF16 blocks into 8 packed E2M1 payload bytes plus 1 UE4M3 scale byte (**`3.5556x`** compression vs BF16: `1,048,576 B -> 294,912 B` for `4096 x 128`), using `rcp.approx.ftz.f32` to match `flashinfer::nvfp4_kv_quantize` and eliminate the **`~0.1%`** nibble drift caused by IEEE division.
- **`MiniMaxM3Nvfp4Config` (#46140)**: Combines `[1, 32]` MXFP8 projections (automatically converted to `[128, 128]` block-FP8 when `not get_platform().is_blackwell`) with `group_size=16` NVFP4 routed experts.
- **Disaggregated KV Checksum Gating (#39729)**: Disables full-prompt KV checksum when `--disaggregation-decode-enable-radix-cache` is active, gates verification on `kv_layout_signature(kv_args)`, and seeds the test mismatch injector per `(random_seed, attn_tp_rank)`.

---

## 3. Reproducible Verification & Summary Matrix

```bash
python experiments/trtllm_qwen3vl_graph_gemma4_vllm_cublas_tail_sglang_msa_nvfp4/run.py --check
python -m unittest tests/test_phase75_experiments.py
```

| Engine / Feature | Core Formula / Mechanism | Baseline / Pre-Fix Behavior | Post-Fix Quantitative Metric |
|---|---|---|---|
| TensorRT-LLM `EncoderGraphIOAdapter` (#20213) | `(total_tokens + 1, num_contexts + 1)` | Padding leaks into real context or 0-token error | Reserves **1 token / 1 context** for isolated dummy padding |
| TensorRT-LLM Gemma4 MM Budgeting (#20663) | `max_soft_tokens * pooling_kernel_size^2` | Under-budgets vision tower by pooled tokens | **`2304`** padded patches/frame (`256 * 3^2`); 4x audio subsampling |
| vLLM Offloading EAGLE Tail (#61019) | `boundary + alignment_or_block` | Keyed attention tail at Mamba `boundary=24` | Maps alignment `4 / 16 / None` to **`28 / 40 / 40`** tokens |
| vLLM ROCm Triton KV View (#60084) | `set_triton_ptr_range` per-layer bound | 32 GiB storage forces 64-bit pointer math | `512 MiB < 2 GiB` per layer restores **32-bit `buffer_load`** |
| SGLang `MSA-v1` NVFP4 KV (#46139) | `16`-elem block + `rcp.approx.ftz.f32` | IEEE divide causes `~0.1%` nibble drift | Zero nibble drift; **`3.5556x`** KV memory compression |
