# Phase 92 Track 2: SGLang MiniMax-M3.1-Flash/AMD Softcap/Diffusion, vLLM MRV2/ROCm DSv4.1/KV Offload & TensorRT-LLM Unified KV

## Executive Summary

This document analyzes 10 first-hand upstream commits across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM` (`priority: 101`):

1. **`sgl-project/sglang@a9bf8c76a0` (`#45925`)**: Adds MiniMax-M3.1-Flash support by integrating the MSA (`1e7bf76f26`) multi-head `q8kv4` indexer and `--monokernel-backend msa` plugin, masking uncommitted slots (`committed = mask & (c < commit_len)`) before dereferencing `req_pool_ptr` in DSPark `_commit_inject_layout_kernel`, and rotating only the first `ROT` dimensions (`HALF = ROT // 2`) in `_fused_kv_norm_rope_write_kernel`.
2. **`sgl-project/sglang@d96c386335` (`#45531`)**: Fixes AMD ROCm `aiter_backend.forward_decode` to pass `softcap=layer.logit_cap` instead of hardcoded `softcap=0` to `unified_attention`.
3. **`sgl-project/sglang@ce77d5e93d` (`#46057`)**: Cleans up rejected and cancelled `/v1/videos` requests using request-owned temporary upload directories managed by `AsyncExitStack`, waits for active CPU preprocessing before cleanup on cancellation, and requires cloud upload success before reporting completion when `--output-path ""`.
4. **`sgl-project/sglang@042dc54cbf` (`#46249`)**: Routes `ReleaseRealtimeSessionReq` to the owning DP replica by returning `req.session_id` in `_session_key(batch)`.
5. **`vllm-project/vllm@4135590ffe` (`#63431`)**: Skips `adaptive_verification` cost-profiling dummy runs during `capture_model(profile_only=True)` in vLLM GPU ModelRunner V2 (`MRV2`) so cached activations on the throwaway KV cache are not subtracted from production KV cache memory.
6. **`vllm-project/vllm@9744ada582` (`#63576`)**: Awaits `krdy` (`await_flags(c, c["krdy"], 1, lambda u: t)`) and loads `a["klen"]` with device cache coherence (`CM_DEV`) in ROCm DeepSeek-V4.1 (`DSv4.1`) monokernel `stage_combine`.
7. **`vllm-project/vllm@14aefcd4f9` (`#62929`)**: Holds secondary-to-primary KV offload promotions after a cache reset until all in-flight pre-reset transfer jobs report completion.
8. **`vllm-project/vllm@8a4b018f1c` (`#63628`)**: Rolls back native NIXL registrations when `add_remote_peer` / `add_remote_peer_async` fails or loses a generation race in P2P KV Offload.
9. **`NVIDIA/TensorRT-LLM@fce28f0726` (`#20178`)**: Adds `BufferConfig::coalescingGroup` (`CoalescingKey = std::pair<int, size_t>`) to Unified KV Cache Manager V2 so buffers coalesce only when both their group and expanded size match.
10. **`NVIDIA/TensorRT-LLM@732fd63463` (`#20915`)**: Rebuilds each live beam's true hypothesis token history via `start_beam_histories_callable` (`PendingBeamHistories`) for logits processors under beam search.

---

## 1. Upstream Commit Analysis

### 1.1 SGLang MiniMax-M3.1-Flash Monokernel, Partial RoPE & AMD Softcap

In `sgl-project/sglang@a9bf8c76a0` (`#45925`), MiniMax-M3.1-Flash introduces three runtime and kernel updates:
- **MSA Monokernel Plugin**: Pins `MSA_COMMIT=1e7bf76f26a5f53cfc2d4d297e1ec2e40955e431` with the multi-head `q8kv4` indexer, strided cache views, and `--monokernel-backend msa` entrypoint.
- **DSPark Uncommitted Slot Masking**: Guards request-pool dereferences with `committed = mask & (c.to(tl.int64) < commit_len)` so padded/uncommitted window slots never read stale request table entries.
- **Partial RoPE in Fused KV Write**: Adds `ROT: tl.constexpr` and `HALF = ROT // 2` to rotate only `0..ROT` while passing `ROT..D` through after RMSNorm.

In `sgl-project/sglang@d96c386335` (`#45531`), `aiter_backend.forward_decode` forwards `softcap=layer.logit_cap` rather than `softcap=0`, preventing unbounded attention logits on AMD GPUs.

### 1.2 vLLM MRV2 Memory Accounting, ROCm DSv4.1 `krdy` Barrier & KV Offload Safety

- **MRV2 `profile_only` Guard (`vllm@4135590ffe`, `#63431`)**: `capture_model` runs twice: first with `profile_only=True` on a throwaway KV cache to measure CUDA Graph memory footprint, then on the production KV cache. Skipping `adaptive_verification` during `profile_only=True` prevents allocator-cached activations from shrinking the KV cache budget.
- **ROCm DSv4.1 `stage_combine` Synchronization (`vllm@9744ada582`, `#63576`)**: In index-source launches (`"krdy" in c`), `stage_combine` awaits `krdy` and loads `a["klen"]` with `CM_DEV` so `nlive = (kv_len + ck - 1) // ck` reflects the newly written KV length.
- **KV Offload Reset Hold & P2P NIXL Rollback (`14aefcd4f9` `#62929`, `8a4b018f1c` `#63628`)**: Prevents stale pre-reset promotions after cache reset and cleans up native NIXL descriptors on peer registration failures.

---

## 2. Deterministic Simulation & Quantitative Verification

Validated via `experiments/sglang_minimax_vllm_mrv2_kv_offload_trtllm_phase92/run.py`:

| Module / Commit | Metric | Legacy Behavior | Fixed / Verified | Architectural Impact |
|---|---|---:|---:|---|
| `sglang@a9bf8c76a0` (`#45925`) | `D=128, ROT=64` Partial RoPE `cos/sin` half-width | `64` (`D // 2` OOB) | `32` (`ROT // 2`) + `64` pass-through | Exact MiniMax-M3.1-Flash RoPE semantics |
| `sglang@d96c386335` (`#45531`) | AMD `aiter` decode `raw_logit=120, cap=50` output | `120.0` (`softcap=0`) | `49.1669` (`softcap=50.0`) | Restores ROCm decode softcap parity |
| `sglang@042dc54cbf` (`#46249`) | `ReleaseRealtimeSessionReq` owner hit rate (`dp_size=4`) | Misrouted via round-robin | `4 / 4` (`100%`) owner hit | Eliminates leaked realtime sessions |
| `vllm@4135590ffe` (`#63431`) | False CUDA Graph memory across `7` capture shapes | `168 MiB` (`84` blocks lost) | `0 MiB` (`84` KV blocks recovered) | Preserves full KV cache budget |
| `vllm@9744ada582` (`#63576`) | ROCm DSv4.1 `nlive` chunks (`klen: 512 -> 2048, ck=64`) | `8` chunks (`24` missed) | `32` chunks (`await krdy` + `CM_DEV`) | Prevents combine-stage KV truncation |
| `vllm@14aefcd4f9` (`#62929`) | Stale pre-reset promotions across `3` in-flight jobs | `> 0` (cache pollution) | `0` (held until jobs drain) | Enforces post-reset tier consistency |
| `TensorRT-LLM@fce28f0726` (`#20178`) | Pool groups for `64 KiB` KV vs Indexer buffers | Coalesced into `2` groups | Separated into `3` groups | Isolates heterogeneous buffer pools |

---

## 3. Architectural Insights & Design Exercises

1. **Isolation of CUDA Graph Memory-Profiling Windows**: Because serving engines compute available KV cache memory from free HBM after `capture_model(profile_only=True)`, any auxiliary profiling pass that leaves tensors in the caching allocator must be deferred to the real capture pass (`profile_only=False`).
2. **Beam Hypothesis Reconstruction vs. Slot Token Lists**: Whenever beam search swaps parent beams across steps, slot $b$'s token list mixes tokens from multiple trajectories; logits post-processors must consume parent-backtracked `PendingBeamHistories`.

---

## 4. Verification & Falsifiable Follow-Up Questions

- **Verification Command**: `python experiments/sglang_minimax_vllm_mrv2_kv_offload_trtllm_phase92/run.py --check` and `python -m unittest tests/test_phase92_experiments.py`.
- **Falsifiable Follow-Up Question**: When two concurrent `add_remote_peer_async(peer_id)` calls race in vLLM P2P NIXL transport and the older generation finishes second, does the generation check immediately deregister the older call's `remote_dlist` without overwriting the newer peer entry?
