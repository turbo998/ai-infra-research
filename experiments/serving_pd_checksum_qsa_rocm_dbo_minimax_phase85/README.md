# Serving PD KV Checksum, QSA Prefill Gather, EAGLE HiCache Restore, ROCm 32-Bit Triton Ptr Range, DBO Microbatches & MiniMax-M3 VL (Phase 85)

Deterministic Python 3.12 standard-library model reproducing the 10 serving-engine upstream changes audited in Phase 85 (`priority: 94`) across SGLang, vLLM, and TensorRT-LLM:

1. **SGLang Disaggregated KV Adler-32 Checksum & Aux-Buffer Guard (`5d11819f`, PR #39729)**:
   - Models positional aux-buffer pairing (`min(len(prefill_aux_ptrs), len(dst_aux_ptrs))`) across Mooncake and MORI RDMA backends, and verifies why enabling both `--disaggregation-enable-kv-checksum` and `--disaggregation-decode-enable-radix-cache` disables whole-prompt Adler-32 verification when decode prefix hits cause prefill to transfer only the suffix.
2. **SGLang Vectorized LTX-2 Diffusion Q/K RMSNorm + Split-RoPE Kernel (`a887ac8f`, PR #46111)**:
   - Models the `kVec=true` template specialization processing `kWidth = 8` consecutive complex pairs (`16` BF16 values = `32 B`, `16-byte` aligned loads/stores) per thread per iteration with identical per-element FMA (`y0 = fmaf(-sin, x1, cos*x0)`, `y1 = fmaf(sin, x0, cos*x1)`), reducing global memory transactions by `8x` over scalar loads.
3. **SGLang Single-Launch QSA Prefill Compressed-Key Gather (`dba22e9b`, PR #45752)**:
   - Models `_gather_qsa_compressed_keys_kernel` launching a 2D Triton grid `(ceil(max_blocks / BLOCK_B), num_seqs)` that computes compressed slots via `(first_raw_slot // RATIO)` on page-aligned QSA allocations, replacing $B$ per-request Python `index_select` + `torch.cat` calls with `1` kernel launch and `0` host-device syncs when `prefill_sequence_lengths_cpu` is present.
4. **SGLang EAGLE Bigram Boundary Token in PD HiCache Decode Restores (`0ff67ee7`, PR #45076)**:
   - Models `_bigram_boundary_tokens(tree_cache) = int(tree_cache.tree_core.is_eagle)` so `rematch_token_len = pm.decode_prefix_len + 1` covers all $N$ bigrams keyed by $N + 1$ token IDs during `DecodePreallocQueue._try_hicache_queue_load_back`.
5. **vLLM ROCm 32-Bit Buffer Addressing for `TRITON_ATTN` KV Cache Views (`a55937f2`, PR #60084)**:
   - Models `set_triton_ptr_range(kv_cache, key_cache, value_cache)` setting `view.ptr_range = region_end - view.data_ptr()` so per-layer KV views inside a $> 2\text{ GiB}$ unified KV cache allocation remain $< 2^{31}\text{ B}$ (`2,147,483,648 B`), restoring 32-bit buffer descriptors (`buffer_load_dword`) on AMD CDNA GPUs instead of falling back to 64-bit scalar+vector pointer arithmetic.
6. **vLLM MRV2 Dual-Batch Overlap (DBO) Routed-Expert Offset & First-Failure Reporting (`2c19f679` PR #63372 & `66bf9315` PR #63476)**:
   - Models `dbo_current_ubatch_token_offset()` indexing `device_buffer[offset : offset + len(local_topk_ids), layer_id]` so microbatch 1 does not overwrite microbatch 0's routed experts (`[0..M_0)` vs `[M_0..M_0+M_1)`), and verifies `next(iter(errors))` vs `min(errors)` so a failure in microbatch 1 is reported directly rather than masked by microbatch 0's subsequent `dbo_register_recv_hook` exception.
7. **vLLM `SimpleCPUOffloadConnector` Chunked-Local Alignment Guard (`65bc6983`, PR #63266)**:
   - Models `num_computed_tokens % spec.attention_chunk_size != 0` rejecting misaligned CPU suffix hits when a partial GPU prefix ends mid-chunk (e.g., `16 % 64 != 0`), preventing suffix-relative chunk boundaries from diverging from absolute token positions in Llama-4 chunked-local attention.
8. **TensorRT-LLM MiniMax-M3 VL Serve-Rendered Prompts, Vision Skip & LoRA `ckpt_source` (`a0cee56d` PR #20626 & `9666d742` PR #20774)**:
   - Models `TextPrompt.chat_template_applied = True` preventing double chat-template wrapping on bare `]<]image[>[` / `]<]video[>[` markers, skipping vision tower forward when all placeholder rows are cached/beyond the active chunk, and preserving per-request `py_lora_ckpt_source` across PeftCache eviction and reload.

## Run

```bash
python experiments/serving_pd_checksum_qsa_rocm_dbo_minimax_phase85/run.py --check
```
