# SGLang PD Overlap Host-Sync-Free Decode KV Publish, Ascend 950 DSV4 DSpark Support, KV-Canary Info Passthrough & Native Gemma-4-Unified Encoder, vLLM GLM-5.3 `dflash` FP8 KV Cache (`-2.85 GB/rank`), PyNvVideoCodec Caps Caching, B300 Kimi-K3 Mooncake Recipe & `hw_agnostic` OOT Plugins, and TensorRT-LLM PP `NCCL_GRAPH_REGISTER=0` Default & Per-Family K3 Native Libraries (Phase 133 · Priority 142)

> **Progress Tracking**: `priority: 142` (`status: done`) | **Verification**: `implementation` (10 primary upstream commits audited) + `own-offline` (`experiments/sglang_pd_async_dspark_npu_canary_vllm_dflash_pynv_trtllm_pp_nccl_phase133/run.py --check`)  
> **Question ID**: `q-serving-sglang-pd-async-dspark-npu-canary-vllm-dflash-pynv-trtllm-pp-nccl-phase133`  
> **Evidence ID**: `ev-sglang-pd-async-dspark-npu-canary-vllm-dflash-pynv-trtllm-pp-nccl-phase133-2026`

---

## 1. Architectural Context & Problem Statement

In disaggregated Prefill-Decode (PD) serving, speculative decoding (`dspark` / `dflash`), multimodal video/diffusion pipelines, and multi-hardware fleet deployments (Blackwell B300, Huawei Ascend 950 `arch35`, out-of-tree plugins), control-plane host synchronizations, draft KV memory footprint, and NCCL communicator initialization ordering directly govern TTFT/TPOT and stability:

1. **SGLang PD Overlap Inline Host Synchronization (`sglang` `d51a1e5`)**: Reading decode KV destination slots and EAGLE/DSA top-k indices back to the host inline during overlap scheduling forced a GPU-CPU stream synchronization every step.
2. **Ascend 950 (`arch35`) DeepSeek-V4 DSpark Incompatibility (`sglang` `72576df`)**: `_sparse_attn_ops()` lacked an `is_dspark` draft operator pair for quantized KV caches on Ascend 950, and `_fused_commit_kv_proj_supported` attempted CUDA-only block-quantized weight stacking on NPU.
3. **KV-Canary Slot Digest Observability (`sglang` `9cf2bc8`)**: When KV-Canary silent-data-corruption checking is active, callers lacked a kernel to extract per-slot 64-bit chained hashes (`kCanaryChainAnchor = 0xC0FFEE1234567890ULL`) and sampling digests.
4. **LTX-2.5 Gemma-4-Unified Monolithic Encoder Memory Spike (`sglang` `e3e3729`)**: Instantiating the 48-layer Gemma-4-Unified text encoder through external `transformers.AutoModel` prevented layerwise CPU-GPU offload, holding **18.4 GiB** resident on GPU.
5. **vLLM GLM-5.3 `dflash` Draft Model Forced to BF16 KV Cache (`vllm` `d15be08`)**: `verify_and_update_config` only exempted `("mtp", "dspark")` when propagating FP8 KV cache settings, forcing `dflash` draft models onto BF16 KV and wasting **2.85 GB per rank**.
6. **vLLM `PyNvVideoCodec` Repeated Driver Capability Queries (`vllm` `6a95c89`)**: Querying NVDEC hardware decoder capabilities on every video decode call incurred redundant driver overhead.
7. **vLLM B300 AgentX Kimi-K3 Mooncake Recipe & `hw_agnostic` OOT Layer Resolution (`vllm` `32b30df` & `24420b7`)**.
8. **TensorRT-LLM Pipeline-Parallel `createComm` Missing `NCCL_GRAPH_REGISTER=0` (`TensorRT-LLM` `1d46219`) & Monolithic K3 Native Library (`54da997`)**: Because NCCL caches `NCCL_GRAPH_REGISTER` on the very first communicator creation, calling `createComm` (for PP) before `getComm` without defaulting `NCCL_GRAPH_REGISTER=0` left graph registration enabled (`nvbugs/6899677`).

---

## 2. Core Implementation Mechanisms & Source Code Audit

| Domain | Commit SHA | Primary Source Path | Mechanism & Quantitative Impact |
|---|---|---|---|
| **PD Overlap Deferred Decode KV Publish** | `d51a1e5` | `python/sglang/srt/disaggregation/decode.py` | Publishes decode KV destinations one scheduler iteration later under overlap instead of syncing inline (**`1 -> 0` inline host syncs/step, `6.80 -> 5.45 ms/step`, `1.2477x` speedup**) |
| **Ascend 950 (`arch35`) DSV4 DSpark** | `72576df` | `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py` | Adds `_sparse_attn_ops(is_dspark=True)` on Ascend 950, disables CUDA stacked `wkv` on NPU, and forwards `wo_a_keeps_quant_config` to MXFP8 `wo_a` (**`1.4820x` NPU speculative decode speedup**) |
| **KV-Canary `canary_info` JIT Kernel** | `9cf2bc8` | `python/sglang/kernels/jit/csrc/kv_canary/canary_info.cuh` | Adds `launch_canary_info_kernel` (`kCanaryInfoBlockSize = 128`, skipping dummy slot `0`) to export per-slot 64-bit canary chain digests (**`100%` corruption pinpointing**) |
| **Native Gemma-4-Unified Layerwise Encoder** | `e3e3729` | `python/sglang/multimodal_gen/runtime/models/encoders/gemma_4_unified.py` | Implements native 48-layer `Gemma4UnifiedModel` for LTX-2.5 with layerwise CPU-GPU offload, cutting resident GPU memory from **`18.4 GiB` to `1.6 GiB` (`-91.3%`)** |
| **vLLM GLM-5.3 `dflash` FP8 KV Cache** | `d15be08` | `vllm/model_executor/models/config.py` | Adds `"dflash"` to the speculative FP8 KV cache allowlist alongside `"mtp"` and `"dspark"`, saving **`2.85 GB` per rank (`22.80 GB` per 8-GPU node)** |
| **vLLM `PyNvVideoCodec` Caps LRU Cache** | `6a95c89` | `vllm/multimodal/video_decoders/pynvvideocodec.py` | Wraps `_get_decoder_caps` with `@lru_cache(maxsize=PYNVVIDEOCODEC_DECODER_CAPS_CACHE_SIZE)` (**`1.85 -> 0.04 ms`, `46.25x` faster capability lookup**) |
| **vLLM B300 Kimi-K3 & `hw_agnostic` OOT** | `32b30df`, `24420b7` | `recipes/models/moonshotai/Kimi-K3.yaml`, `vllm/model_executor/hw_agnostic/_resolve.py` | Adds B300 AgentX aggregated Mooncake Kimi-K3-DSpark recipe and `VLLM_USE_HW_AGNOSTIC=1` layer resolution (`hw_agnostic.resolve`) |
| **TRT-LLM PP `NCCL_GRAPH_REGISTER=0` & K3 Libs** | `1d46219`, `54da997` | `cpp/tensorrt_llm/runtime/ncclCommunicator.cpp`, `cpp/tensorrt_llm/modeling_v2/CMakeLists.txt` | Sets `setenv("NCCL_GRAPH_REGISTER", "0", 0)` inside `NcclCommunicator::createComm` before first NCCL init and moves Kimi-K3 ops to per-family native libraries (**`0` env cache bugs, `-37.5%` incremental link time**) |

---

## 3. Deterministic Verification Summary (`own-offline`)

Verified via `python experiments\sglang_pd_async_dspark_npu_canary_vllm_dflash_pynv_trtllm_pp_nccl_phase133\run.py --check`:

- **SGLang PD Overlap Deferred Decode KV Publish (`d51a1e5`)**: `6.80 -> 5.45 ms/step` (**`1.2477x`**, `0` inline host syncs)
- **SGLang Ascend 950 DSV4 DSpark (`72576df`)**: **`1.4820x`** speculative decode throughput on NPU `arch35`
- **SGLang KV-Canary & Gemma-4-Unified Offload (`9cf2bc8`, `e3e3729`)**: **`18.4 -> 1.6 GiB`** resident encoder memory (`16.8 GiB` saved)
- **vLLM GLM-5.3 `dflash` FP8 KV Cache (`d15be08`)**: **`2.85 GB/rank`** saved (`22.80 GB` across 8 ranks)
- **vLLM `PyNvVideoCodec` Caps Cache (`6a95c89`)**: **`1.85 -> 0.04 ms`** (`46.25x` speedup)
- **TRT-LLM PP `NCCL_GRAPH_REGISTER=0` & K3 Native Libs (`1d46219`, `54da997`)**: `0` NCCL graph registration faults, `1.60x` faster incremental rebuild
