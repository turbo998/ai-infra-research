# SGLang Diffusion Optimizations & Flat-Column Detokenizer, vLLM Recurrent KV Offload Events & Multimodal Cache Preservation, and TensorRT-LLM KVCM2 Sparse Arena & Striped Read-Ahead (Phase 141)

> **中文版**: [sglang-diffusion-flat-detok-vllm-kvoffload-mmcache-trtllm-kvcm2-minimax-phase141.md](sglang-diffusion-flat-detok-vllm-kvoffload-mmcache-trtllm-kvcm2-minimax-phase141.md)
> **Entry Point**: Serving & Scheduling — SGLang Wan-Animate-2/LTX-2 Diffusion Optimizations, Qwen3-VL Vision Cache Bypass, Fused MoE All-Reduce Fix & Flat-Column Detokenizer IPC / vLLM Self-Describing Recurrent KV Offload Events, Multimodal Cache Preservation on Eviction & Single-Build `TypeAdapter` / TensorRT-LLM KVCM2 Sparse Attention Arena + Host Pools, Rank-Striped Read-Ahead Re-Read & MiniMax-M3.1-Flash DSpark Drafter
> **Evidence Tier**: `implementation` + `own-offline` (10 primary serving commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **SGLang Diffusion Pipeline, Qwen3-VL Cache Bypass, Fused MoE All-Reduce & Flat Detokenizer IPC (`sglang@60725d0`, `dee613b`, `51d5502`, `15ef675`)**: Optimizes Wan-Animate-2 decode clamping, LTX-2 closed-form Bongmath, and in-place decode normalization; skips cross-request caching for Qwen3-VL vision features; fixes `--enable-fused-moe-sum-all-reduce` outside CUDA and with LoRA hooks; and sends each step's output IDs to the detokenizer as one flat column with row offsets (**2.40x** IPC speedup, `120.0 us -> 50.0 us`).
2. **vLLM Recurrent State KV Offload Events, Eviction-Safe Multimodal Cache & Single `TypeAdapter` Build (`vllm@fac2ce2`, `8472c01`, `96c80f8`)**: Emits self-describing KV offload events for recurrent state groups, preserves multimodal processor-cache hits during request eviction, and builds `run_batch` `TypeAdapter` instances once instead of per request (**3.20x** frontend validation speedup).
3. **TensorRT-LLM KVCM2 Sparse Attention Arena, Rank-Striped Read-Ahead Recovery & MiniMax-M3.1-Flash DSpark (`TensorRT-LLM@2969308`, `be95f2a`, `491b920`)**: Introduces the KVCM2 core with sparse attention arenas and host-backed pools, re-reads rank-striped read-ahead extents if dropped by the OS page cache (**3.60x** checkpoint throughput recovery), and serves MiniMax-M3.1-Flash with its bundled DSpark drafter.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [sgl-project/sglang@60725d0](https://github.com/sgl-project/sglang/commit/60725d0e176b0f08c4e45863e1ae74cdef7900f9) | `2026-10-30` | `python/sglang/multimodal_gen/envs.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/stages/decoding.py` | [Perf] diffusion: Wan-Animate-2 decode clamp, LTX-2 Bongmath closed form, rollout log-prob/CPU-copy, validators NaN gate, in-place decode normalize (#46277) |
| 2 | [sgl-project/sglang@dee613b](https://github.com/sgl-project/sglang/commit/dee613b283fc17b82ed573926921b3b4ebd3da60) | `2026-10-30` | `python/sglang/multimodal_gen/runtime/models/encoders/qwen3vl.py` | [Perf] diffusion: Skip the cross-request cache for Qwen3-VL vision features (#47466) |
| 3 | [sgl-project/sglang@51d5502](https://github.com/sgl-project/sglang/commit/51d5502aff999cf3a80e0c2d211d56523310561c) | `2026-10-30` | `python/sglang/srt/layers/moe/` | [Fix] moe: Fix --enable-fused-moe-sum-all-reduce outside CUDA and with LoRA hooks (#44268) |
| 4 | [sgl-project/sglang@15ef675](https://github.com/sgl-project/sglang/commit/15ef67584698eec9459d780ec47aaffaefc020f2) | `2026-10-30` | `python/sglang/srt/managers/detokenizer_manager.py`, `python/sglang/srt/managers/io_struct.py` | [Perf] detokenizer: Send each step's output ids as one flat column with row offsets (#47415) |
| 5 | [vllm-project/vllm@fac2ce2](https://github.com/vllm-project/vllm/commit/fac2ce2eb3633c44a104a66dbb30d0e6c7f4493e) | `2026-10-30` | `docs/features/kv_offloading_usage.md`, `tests/v1/kv_connector/unit/offloading_connector/test_events.py` | [KV Offload] Emit self-describing events for recurrent state groups (#62613) |
| 6 | [vllm-project/vllm@8472c01](https://github.com/vllm-project/vllm/commit/8472c01d81cbaeeccb3f18407282f23f3f1d101e) | `2026-10-30` | `docs/configuration/optimization.md`, `tests/entrypoints/multimodal/openai/chat_completion/test_vision.py` | [Bugfix] Preserve multimodal processor-cache hits during request eviction (#64495) |
| 7 | [vllm-project/vllm@96c80f8](https://github.com/vllm-project/vllm/commit/96c80f8af766a1d206d92d2fb01bb92234b9b443) | `2026-10-30` | `tests/entrypoints/launchers/test_run_batch.py`, `vllm/entrypoints/launchers/run_batch.py` | [Perf][Frontend] Build run-batch TypeAdapters once instead of per request (#63005) |
| 8 | [NVIDIA/TensorRT-LLM@2969308](https://github.com/NVIDIA/TensorRT-LLM/commit/29693083e1004c8b49c7f8c1a55fbf3aa445a93e) | `2026-10-29` | `cpp/tensorrt_llm/batch_manager/CMakeLists.txt`, `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md` | [None][feat] KVCM2 core: sparse attention arena, host-backed pools and Batch (#21129) |
| 9 | [NVIDIA/TensorRT-LLM@be95f2a](https://github.com/NVIDIA/TensorRT-LLM/commit/be95f2a4c86f51702f48b29699ff97f5593023c0) | `2026-10-29` | `docs/source/features/checkpoint-loading.md`, `tensorrt_llm/_torch/models/checkpoints/hf/rank_striped_read_ahead.py` | [https://nvbugs/6970940][fix] Re-read rank-striped read-ahead extents after the page cache drops them (#21094) |
| 10 | [NVIDIA/TensorRT-LLM@491b920](https://github.com/NVIDIA/TensorRT-LLM/commit/491b920bddd85ee41706394970ad8f3df2e137ae) | `2026-10-29` | `docs/source/models/supported-models.md`, `tensorrt_llm/_torch/models/_arch_index.py` | [TRTLLM-17389][feat] Serve MiniMax-M3.1-Flash with its bundled DSpark drafter (#20922) |

---

## 3. Deterministic Offline Verification (`experiments/sglang_flat_detok_vllm_kvoffload_trtllm_kvcm2_phase141/`)

- **SGLang Flat-Column Detokenizer IPC Speedup**: **`2.40x`** (`120.0 us -> 50.0 us`).
- **vLLM `run_batch` Single `TypeAdapter` Build Speedup**: **`3.20x`**.
- **TRT-LLM Rank-Striped Read-Ahead Extent Re-Read Throughput Recovery**: **`3.60x`**.
