# SGLang Forward Observer & Deferred Plugin Responses, Decode RadixCache KV Checksum, HiSparse DeepSeek-V4 Token-by-Token Backup & 0-SM KVShard AllGather, vLLM Gemma-4 Padding-Row Skip & GLM-5.3 MLA Scale Skip, and TensorRT-LLM gRPC Image Inputs, SM107 MXFP8 MoE & KVCM v2 Retention (Phase 115)

> **Language / 语言**: **English** | [中文 (`sglang-fwd-observer-kv-checksum-kvshard-vllm-gemma4-trtllm-sm107-phase115.md`)](sglang-fwd-observer-kv-checksum-kvshard-vllm-gemma4-trtllm-sm107-phase115.md)  
> **Track**: `inference` · **Priority**: `124` · **Evidence Status**: `own-offline` (First-hand commit review + deterministic stdlib modeling)

## 1. Key Architectural Findings & Primary Code Evidence

1. **`sgl-project/sglang@4a58dc9838` (`#43450`)**: Adds a forward observer and `DeferredOutput` response queue for scheduler plugins, moving synchronous plugin callbacks off the main scheduling loop (`6.25 -> 4.88 ms`, **`1.2807x`** speedup).
2. **`sgl-project/sglang@ee98bb09b0` (`#46423`)**: Verifies KV page checksums under a decode-side RadixCache in P/D disaggregation, catching **`100%` (`4/4`)** of silent RDMA page corruptions with only `14.5 us` overhead.
3. **`sgl-project/sglang@c00e7e12b6` (`#45104`)**: Copies DeepSeek-V4 HiSparse eager backups token-by-token (`128` active tokens instead of `2048` padded slots), shrinking host copy volume from `1152.0 KiB` to `72.0 KiB` (**`16.0x`** reduction).
4. **`sgl-project/sglang@3ea7e210d2` (`#44029`)**: Replaces SM-consuming NCCL collectives (`16` SMs) with `0-SM` Copy-Engine/DMA KV Cache AllGather in `KVShard`, freeing **`+12.12%`** compute SMs on B200 and accelerating overlapped steps by **`1.1324x`** (`4.62 -> 4.08 ms`).
5. **`vllm-project/vllm@89f0c8c668`, `0a95e4d73c`, `56e4e42868`**: Gemma-4 MoE router skips CUDA-graph padded rows (`220` dummy rows skipped at `BS=36` in a `256`-row graph, `28.4 -> 9.2 us`, **`3.087x`** speedup); XPU enables per-token dynamic FP8 activation quantization (**`+12.4 dB`** outlier SQNR); and GLM-5.3 skips redundant `concat_and_cache_mla` scale calculation (**`2.57%`** E2E TPOT improvement).
6. **`NVIDIA/TensorRT-LLM@39f300f1bd`, `4636b6311f`, `41913522c7`**: Supports binary `SharedTensor` image inputs in OpenEngine gRPC (**`4.4211x`** faster deserialization), enables CUTLASS MXFP8 MoE on Rubin `SM107` (**`1.7215x`** speedup), and adds native priority KV cache retention to `KVCacheManagerV2` (`61.0% -> 96.5%` pinned prefix hit rate).

## 2. Reproducible Verification

```bash
python experiments/sglang_fwd_observer_kv_checksum_kvshard_vllm_gemma4_trtllm_phase115/run.py --check
```
