# SGLang Disaggregated Decode Radix Cache Incremental KV Checksum, LTX-2 Vectorized Fused RoPE, vLLM MLA Unit-Scale Skip, Shared NCCL RL Weight Rendezvous, and TensorRT-LLM MiniMax-M3 Piecewise Graph Fix

**中文版**: [sglang-disagg-radix-checksum-ltx2-rope-vllm-mla-scale-skip-nccl-rl-trtllm.md](sglang-disagg-radix-checksum-ltx2-rope-vllm-mla-scale-skip-nccl-rl-trtllm.md)

## 1. Context and Core Architectural Problems

Modern disaggregated LLM serving, diffusion video generation, FP8 Multi-head Latent Attention (MLA), and online RL weight synchronization face five critical correctness and tail-latency bottlenecks:

1. **False-Positive KV Checksum Failures with Decode-Side Radix Cache in SGLang**: Previously, `--disaggregation-enable-kv-checksum` digested the full prompt `[0, end_idx)`, whereas with `--disaggregation-decode-enable-radix-cache` prefill only transfers suffix pages `[start_idx, end_idx)` past the cached prefix, forcing SGLang to disable KV checksums whenever decode radix caching was active.
2. **Scalar Memory Access Bottleneck in LTX-2 Q/K RMSNorm + Split-RoPE**: Processing one complex RoPE pair per loop iteration underutilizes 16-byte vector load/store bandwidth on Blackwell GPUs.
3. **Redundant Scale Multiplication in vLLM MLA FP8 Cache**: When `mla_k_scale == 1.0`, `concat_and_cache_mla_kernel` still dereferenced `*scale` from global memory and multiplied every element across all 576 MLA dimensions.
4. **Duplicated and Deadlock-Prone NCCL Trainer Rendezvous in vLLM RL Weight Transfer**: Dense and sparse NCCL weight transfer engines duplicated worker payload construction and required concurrent execution of trainer endpoint creation and worker process-group initialization.
5. **KV Cache Corruption in TensorRT-LLM MiniMax-M3 Piecewise CUDA Graphs**: When `_build_msa_fields()` ran before restoring `metadata._num_tokens = live_tokens`, `msa_out_cache_loc` generated write indices for padded tokens and overwrote subsequent KV cache slots.

---

## 2. First-Hand Source Code Audit

### 2.1 SGLang Suffix-Sliced Adler-32 KV Checksum with Decode Radix Cache (`#46423`, commit `ee98bb09b028`)

In `python/sglang/srt/disaggregation/{checksum,prefill,decode}.py`:
- `page_indices_for_request(scheduler, req, end_idx, start_idx=0)` and `state_indices_for_request` now slice `req_to_token[req.kv.req_pool_idx, start_idx:end_idx]` where `start_idx = req.disagg_decode_prefix_len`.
- Both prefill (`send_kv_chunk`) and decode (`_verify_kv_checksums`) compute the Adler-32 digest strictly over the transferred suffix pages `[start_idx, end_idx)`, enabling end-to-end KV corruption detection alongside decode-side radix caching.

### 2.2 SGLang LTX-2 8-Pair (16-Byte) Vectorized Q/K RMSNorm + Split-RoPE (`#46111`, commit `a887ac8fc6ea`)

In `python/sglang/kernels/kda_kernels/csrc/diffusion/ltx2_qknorm_split_rope.cuh`, adding template parameter `bool kVec` processes `kWidth = 8` consecutive pairs per thread via 16-byte accesses while preserving exact scalar FMA rounding, reducing per-token global memory transactions by **8.0x** (`4,096 -> 512` transactions/token at `32` heads, `head_dim=128`). Commit `933853a3f263` (`#45733`) generalizes exact GPU tensor input preprocessing across Flux, Qwen-Image, and Cosmos3.

### 2.3 vLLM `concat_and_cache_mla_kernel` Compile-Time `skip_scale` Specialization (`#62102`, commit `56e4e42868b4`)

In `csrc/cache_kernels.cu`, `concat_and_cache_mla_kernel<scalar_t, cache_t, kv_dt, skip_scale>` introduces an `else if constexpr (skip_scale)` branch that passes literal `1.0f` to `fp8::scaled_convert`, avoiding 576 global `*scale` dereferences and multiplications per token on GLM-5.3 and DeepSeek-V3.2 and yielding a **1.2%–3.0% E2E TPOT improvement**. Complementing this, `#63495` (`0765565f103d`) replaces multi-tensor PyTorch CPU index building in `clef_joint_schema.py` with vectorized NumPy arrays, and `#63496` (`8e11a8a49e11`) rejects incompatible MegaMoE backends when LoRA is enabled.

### 2.4 vLLM Shared NCCL Trainer Rendezvous for Dense and Sparse RL Weight Transfer (`#63460`, commit `c9dbc92630bc`)

In `vllm/distributed/weight_transfer/nccl_common.py`, `build_worker_init_payload` places workers at `rank_offset = 1` after the trainer sender at `rank 0` (preserving `packed=True` for dense transfers and unpacked defaults for sparse transfers), while `run_trainer_rendezvous` runs trainer communicator creation and worker initialization concurrently via a `ThreadPoolExecutor`.

### 2.5 TensorRT-LLM MiniMax-M3 Piecewise CUDA Graph `live_tokens` Restoration (`#19721`, commit `f1d8a73f640f`)

Setting `metadata._num_tokens = live_tokens` prior to `metadata._build_msa_fields()` ensures `msa_out_cache_loc` spans only `[128, 128 + live_tokens)` (e.g., 19 slots instead of 64 padded slots), preventing 45 out-of-bounds KV cache slot overwrites when an ADP rank has padded piecewise graphs. Commit `87342f447486` (`#20270`) narrows `DecoderRunner` / `NoKVCacheRunner` internals to `_prepare_inputs` and `_model_forward`.

---

## 3. Reproducible Verification Summary

Verified via `experiments/sglang_disagg_radix_checksum_ltx2_rope_vllm_mla_scale_skip_nccl_rl_trtllm/run.py`:

| Engine & Subsystem | Mechanism | Unfixed / Baseline | Fixed / Optimized | Evidence Class |
|---|---|---|---|---|
| SGLang Disagg KV Checksum (`#46423`) | Suffix slice `[start_idx, end_idx)` Adler-32 | Full-row mismatch (checksum disabled) | Exact match on 5 transferred pages (`0x6708044d`) | `implementation` + `own-offline` |
| SGLang LTX-2 Split-RoPE (`#46111`) | `kWidth=8` pairs (16B vectorized access) | 4,096 scalar txns/token | 512 vector txns/token (**8.0x** reduction) | `implementation` + `own-offline` |
| vLLM MLA `skip_scale` (`#62102`) | Compile-time `skip_scale=True` (`1.0f`) | 576 `*scale` loads & muls/token | **1.2%–3.0%** E2E TPOT improvement | `author-benchmark` + `own-offline` |
| vLLM NCCL RL Rendezvous (`#63460`) | Concurrent `run_trainer_rendezvous` | Duplicated / deadlock-prone sequential init | Unified `rank_offset=1` concurrent handshake | `implementation` + `own-offline` |
| TRT-LLM MiniMax-M3 (`#19721`) | `metadata._num_tokens = live_tokens` | Overwrites 45 extra KV slots (`64` vs `19`) | Exact 19-slot write range `[128, 147)` | `implementation` + `own-offline` |
