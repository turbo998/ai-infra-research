# SGLang PD Aux-Lengths Frame Alignment & MXFP8 L3 Multi-Buffer HiCache, ROCm Ulysses Head-Group Pipeline, vLLM Batch-Invariant Kernel Pinning & NIXL Layer-Paired KV, and TensorRT-LLM Immediate Cache Reset (Phase 158)

> **中文版**: [sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158.md](sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158.md)
> **Question ID**: `q-sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-phase158` (`priority: 167`)
> **Evidence ID**: `ev-sglang-pd-aux-mxfp8-ulysses-vllm-nixl-pcp-trtllm-2026-phase158`

---

## 1. Architectural Context & Core Findings

Across disaggregated Prefill-Decode (PD) engines, L3 hierarchical KV caches (HiCache), and heterogeneous pipeline/hybrid-attention topologies, control-plane ZMQ frame shifts, missing microscaling (`UE8M0`) scale buffers, bucket-dependent kernel autotuning, and index-based NIXL region pairing cause silent corruption or multi-rank hangs. Auditing 10 primary upstream commits merged on `2026-10-31`, we establish five core architectural findings:

1. **SGLang PD Registration `aux_item_lens` Frame Alignment (`sglang@5eb7018`, `#47638`)**: Moves `dst_aux_item_lens` to ZMQ frame `msg[6]` immediately after `dst_aux_ptrs` (`msg[5]`) across Mooncake and NIXL bootstrap receivers, eliminating frame-offset shifts when optional trailing fields are omitted.
2. **SGLang HiCache L3 MXFP8 Dual-Buffer KV Persistence (`sglang@9780c52`, `#46936`)**: Extends `HostKVCache.get_page_buffer_meta` and L3 storage backends to persist MXFP8 KV pages as multi-buffer objects (`131,072 B` FP8 payload + `4,096 B` `UE8M0` scale buffer = `135,168 B` per 64-token 8-head page), achieving a **1.9394x** compression ratio over BF16 (`262,144 B`).
3. **SGLang ROCm MiniMax-H3 Ulysses Head-Group Pipeline & SM100 Cake Softmax (`sglang@e93f817`, `#47567`; `sglang@c0f710f`, `#44168`)**: Overlaps RCCL All-to-All head-group exchange on a dedicated comm stream with `aiter` dense attention (**1.28x** modeled 2-group speedup, `9.60 -> 7.50 ms`), and enables `cake_blackwell_softmax` on NVIDIA SM100 (`(10, 0)`).
4. **vLLM Batch-Invariant FP8/NVFP4 Pinning, PP>1 Warmup Guard, NIXL Layer Pairing & DSv4.1 PCP (`vllm@54beee8`, `#62192`; `vllm@4aba9bf`, `#64338`; `vllm@8b55f91`, `#63416`; `vllm@2b009fc`, `#63098`)**: Pins FlashInfer `bmm_fp8` (`backend="cutlass"`) and `W4A16` NVFP4 `gemm_fp4_nt` across $M$ buckets under `vllm_is_batch_invariant()` (`0.0 ULP` batch-size drift); fixes `PP > 1` warmup hangs; pairs NIXL KV regions by layer name instead of positional index; and enables DeepSeek V4.1 PCP with `PP > 1` / `DP > 1`.
5. **TensorRT-LLM Immediate `addAllBlocksCleared()` Publication & `LlmRequest` Cleanup (`TensorRT-LLM@17bfe4e`, `#21142`; `TensorRT-LLM@d5752a1`, `#21036`)**: Publishes `addAllBlocksCleared()` immediately inside `BlockRadixTree::clear()` without waiting for the next inference step (`0.0 ms` event lag) and removes dead `promptEmbeddingTable` / `promptVocabSize` fields from `GenericLlmRequest`.

---

## 2. Primary Upstream Evidence Matrix

| Commit | Module / File | Verified Mechanism & Metric |
| :--- | :--- | :--- |
| `sgl-project/sglang@5eb7018` (`#47638`) | `python/sglang/srt/disaggregation/mooncake/conn.py`, `nixl/conn.py` | Places `dst_aux_item_lens` at `msg[6]` right after `dst_aux_ptrs` (`msg[5]`), `0` truncation errors |
| `sgl-project/sglang@9780c52` (`#46936`) | `python/sglang/srt/mem_cache/memory_pool_host.py`, `storage/hf3fs/` | Stores MXFP8 KV pages in L3 as `131,072 B` FP8 + `4,096 B` `UE8M0` scale (`135,168 B`), **1.9394x** compression |
| `sgl-project/sglang@e93f817` (`#47567`) | `python/sglang/multimodal_gen/runtime/layers/attention/minimax_h3_attn.py` | Pipelined Ulysses head-group RCCL All-to-All + attention overlap (**1.28x** speedup) |
| `sgl-project/sglang@c0f710f` (`#44168`) | `python/sglang/srt/layers/sampler.py` | Enables `cake_blackwell_softmax` on `(10, 0)` (SM100) in addition to `(10, 3)` |
| `vllm-project/vllm@54beee8` (`#62192`) | `vllm/model_executor/layers/batch_invariant.py` | Pins `bmm_fp8` and `W4A16` NVFP4 kernels under batch invariance (**0.0 ULP** drift) |
| `vllm-project/vllm@4aba9bf` (`#64338`) | `vllm/v1/worker/gpu_worker.py` | Fixes `_warmup_flashinfer_attention` collective deadlock under `PP > 1` |
| `vllm-project/vllm@8b55f91` (`#63416`) | `vllm/distributed/kv_transfer/kv_connector/v1/nixl_connector.py` | Pairs NIXL KV regions by layer identity across heterogeneous PP/HMA (`0` mismatches) |
| `vllm-project/vllm@2b009fc` (`#63098`) | `vllm/model_executor/models/deepseek_v2.py` | Enables DeepSeek V4.1 Prefill Context Parallelism (PCP) with `PP > 1` or `DP > 1` |
| `NVIDIA/TensorRT-LLM@17bfe4e` (`#21142`) | `cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp` | `BlockRadixTree::clear()` emits `addAllBlocksCleared()` immediately (`0.0 ms` delay) |
| `NVIDIA/TensorRT-LLM@d5752a1` (`#21036`) | `cpp/include/tensorrt_llm/batch_manager/llmRequest.h` | Removes unused `mPromptEmbeddingTable` and `mPromptVocabSize` fields |

---

## 3. Reproducible Verification

```powershell
python experiments\sglang_pd_aux_mxfp8_ulysses_vllm_nixl_pcp_trtllm_phase158\run.py --check
```
