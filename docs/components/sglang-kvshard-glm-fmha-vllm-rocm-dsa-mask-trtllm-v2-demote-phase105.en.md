# SGLang GLM DSA/MTP `KVShard` & Non-Interleaved `fmha_v2` Prefill, vLLM ROCm Single-Launch DSV4.1 Candidate Mask, and TensorRT-LLM Qwen Vision RoPE & V2 Long-Prefill Demotion (Phase 105)

> **Language / 语言**: [中文 (Chinese)](sglang-kvshard-glm-fmha-vllm-rocm-dsa-mask-trtllm-v2-demote-phase105.md) | **English (英文版)**

## 1. Architectural Context & Core Bottlenecks

Across disaggregated Prefill/Decode serving for `GLM-5.x` / `DeepSeek-V4.1` Dynamic Sparse Attention (DSA) + Multi-Token Prediction (MTP), SM90/SM120 GQA prefill, ROCm sparse decoding, and mixed short/long prompt scheduling, five serving bottlenecks arise:
1. **Replicated GLM Latent KV & DSA Indexer Pages on Prefill Workers**: Without page-level sharding, all 8 Attention-CP/TP ranks replicate both target/draft latent KV (`576 B/tok`) and DSA indexer pages (`132 B/tok`, totaling `708 B/tok`).
2. **Temporary KV Interleaving in SM90/SM120 `fmha_v2` & Blocked BF16 MoE Refit Corruption**: `trtllm_mha_backend` copied separate `k_cache`/`v_cache` pages into a temporary interleaved buffer (**`32.0 MiB/layer` at `8,192` tokens**), while RL weight updates overwrote `128x128` blocked TRT-LLM BF16 expert weights without first restoring canonical shapes.
3. **Two-Launch ROCm DSV4.1 Candidate Mask & LDS-Only Barrier Semantics**: On AMD GPUs, `tl.debug_barrier` only flushes LDS traffic, requiring two kernel launches unless global flag stores are explicitly fenced with `s_waitcnt vmcnt(0)`.
4. **Separate Q/K Vision RoPE Launches & Uncached Long-Prefill Head-of-Line Blocking in TensorRT-LLM**: Qwen2-VL launched separate Q and K RoPE kernels across 32 ViT layers (`64` launches), while FIFO V2 scheduling stalled short requests behind long uncached prefills.

## 2. Primary Source Implementation & Mathematical Invariants

- **SGLang GLM `KVShard` (`8136cbce3d`), Non-Interleaved `fmha_v2` (`1a120754e0`), BF16 MoE Refit (`8353f86f0b`) & AMD `gfx1250` FlyDSL (`cf04168be1`)**: Shards `708 B/tok` across 8 prefill CP/TP ranks (**`88.5 B/tok/rank`, `8.0x` unique KV capacity**) with CPU HiCache ownership preservation; passes pre-expanded `[B, 2, M]` `fmha_v2_block_tables` directly to eliminate **`32.0 MiB/layer` copy traffic (`64 -> 0` interleave launches)**; and calls `restore_weights_before_loading` before RL weight refit.
- **vLLM ROCm Single-Launch DSA Mask (`228b1e3d23`), Hybrid Mamba Spec Slots (`e3ba8d00db`) & EAGLE Quantized Head Guard (`43111fd1f0`, `a9b727a612`)**: Inserts inline `s_waitcnt vmcnt(0)` in `_mask_candidates_fused_kernel` (`2 -> 1` launch, **`19.6 -> 10.4 us`, `1.8846x` speedup**), reserves `1 + spec.num_speculative_blocks` (`1 -> 4`) Mamba slots, and guards `_should_share` when `target` uses packed quantized weights.
- **TensorRT-LLM Qwen One-Launch Vision RoPE (`55f7033f71`) & V2 Long-Prefill Demotion (`27c339ab53`)**: Fuses Q/K vision RoPE via `packed_qk` (`64 -> 32` ViT launches, **`19.2 -> 7.8 us`, `2.4615x` speedup**) and demotes requests whose `py_uncached_prefill_tokens > ctx_demote_uncached_tokens` with bounded aging, cutting short-request P95 TTFT from **`285.0 -> 42.0 ms` (`6.7857x`)**.

## 3. Deterministic Verification Summary

Run `python experiments/sglang_kvshard_glm_fmha_vllm_rocm_dsa_trtllm_phase105/run.py --check`:

| Metric | Baseline | Phase 105 | Improvement | Evidence Class |
|---|---:|---:|---:|---|
| GLM DSA/MTP Prefill Storage (`CP=8`) | `708.0 B/tok` | `88.5 B/tok` | **`8.0x` unique KV capacity** | `implementation` + `own-offline` |
| SM90/SM120 `fmha_v2` Interleave Copy (`8,192` tok) | `32.0 MiB/layer` (`64` kernels) | `0.0 MiB/layer` (`0` kernels) | **`100%` copy & launch elimination** | `implementation` + `own-offline` |
| vLLM ROCm DSV4.1 Candidate Mask | `19.6 us` (`2` launches) | `10.4 us` (`1` launch) | **`1.8846x` (`s_waitcnt vmcnt(0)`)** | `implementation` + `own-offline` |
| TRT-LLM Qwen Vision RoPE + mRoPE Prep | `19.2 us` (`64` launches) | `7.8 us` (`32` launches) | **`2.4615x`** | `implementation` + `own-offline` |
| TRT-LLM V2 Short-Request P95 TTFT | `285.0 ms` | `42.0 ms` | **`6.7857x` HoL reduction** | `implementation` + `own-offline` |
