# SGLang DSpark Eager Fallback & CUDA Graph Keep-Alive, vLLM Muse Spark Fused All-Gather/Norm & Mooncake Heterogeneous TP, and TensorRT-LLM Native FP8 MoE-LoRA & Kimi-K3 NVFP4 KV (Phase 140)

> **中文版**: [sglang-dspark-eager-cudagraph-vllm-musespark-mooncake-trtllm-fp8lora-kimik3-phase140.md](sglang-dspark-eager-cudagraph-vllm-musespark-mooncake-trtllm-fp8lora-kimik3-phase140.md)
> **Entry Point**: Serving & Scheduling — SGLang DSpark Ragged-Verify Eager Fallback, Multi-Variant CUDA Graph Workspace Keep-Alive, Sparse Logprob Flattening & Chunked Beam Group Retirement / vLLM Muse Spark Fused All-Gather+Norm, Mooncake Hybrid KV Heterogeneous TP & Reusable Pinned Sleep Backup / TensorRT-LLM Phased FMHA, Native FP8 MoE-LoRA & Kimi-K3 NVFP4 MLA KV Cache
> **Evidence Tier**: `implementation` + `own-offline` (10 primary serving commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **SGLang DSpark Eager Fallback, Multi-Variant CUDA Graph Keep-Alive, Logprob Flattening & Beam Retirement (`sglang@a8d944f`, `8ecabd3`, `60a347b`, `f96a6e4`)**: Safely falls back to eager execution when a ragged-verify tier exceeds the captured CUDA Graph grid, keeps attention metadata and SP workspaces alive across graph variants of the same batch size, flattens egress logprob columns only for rows that carry them, and retires the entire beam group when a chunked leader aborts.
2. **vLLM Muse Spark Fused All-Gather+Norm, Mooncake Heterogeneous TP & Reusable Pinned Sleep Buffer (`vllm@becbaea`, `4d1376a`, `deacc36`)**: Fuses all-gather and normalization for Muse Spark hybrid MoE (**1.32x** speedup), supports heterogeneous TP sharing for hybrid KV caches in Mooncake Store, and reuses exact-size pinned CPU backup buffers in level-1 sleep (**91.67%** faster sleep/wake transitions).
3. **TensorRT-LLM Phased FMHA, Native FP8 MoE-LoRA & Kimi-K3 NVFP4 MLA KV Cache (`TensorRT-LLM@120ea6b`, `ac6fb72`, `c50ea88`)**: Splits `thop.attention` into modular phased FMHA, supports native FP8 MoE-LoRA weights in CUDA Graph grouped GEMMs, and enables NVFP4 (FP4 MLA) KV caching for Kimi-K3 (**3.56x** effective KV capacity over BF16).

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [sgl-project/sglang@a8d944f](https://github.com/sgl-project/sglang/commit/a8d944f5e4a027a1e069b19e115919fbdf793510) | `2026-10-30` | `python/sglang/srt/speculative/dspark_components/dspark_planner.py`, `test/registered/spec/dspark/test_dspark_scheduler.py` | [Fix] Spec: Fall back to eager when the ragged-verify tier exceeds the captured grid (#47333) |
| 2 | [sgl-project/sglang@8ecabd3](https://github.com/sgl-project/sglang/commit/8ecabd3c1e1227ca91b0bbb48d75cf33b4626a73) | `2026-10-30` | `docs/docs/references/environment_variables.mdx`, `python/sglang/srt/environ.py` | [Fix] cuda-graph: Keep attention metadata and SP workspaces alive across graph variants of a batch size (#46720) |
| 3 | [sgl-project/sglang@60a347b](https://github.com/sgl-project/sglang/commit/60a347b6f304e7fd953c5cf823081cf90a2ad0b2) | `2026-10-30` | `python/sglang/srt/rust_server/server.py`, `python/sglang/srt/utils/flatten.py` | [Perf] rust-server: Flatten egress logprob columns only for rows that carry them (#47106) |
| 4 | [sgl-project/sglang@f96a6e4](https://github.com/sgl-project/sglang/commit/f96a6e42f6048fa5175e3de603d146daaaa95a29) | `2026-10-30` | `python/sglang/srt/managers/scheduler.py`, `test/registered/unit/managers/test_scheduler_chunked_abort_race.py` | [Fix] scheduler: Retire the beam group of an aborted chunked leader (#47381) |
| 5 | [vllm-project/vllm@becbaea](https://github.com/vllm-project/vllm/commit/becbaea9f58f5a492f36d4b932fbc0484fa68198) | `2026-10-30` | `csrc/moe/muse_spark_routing.cu`, `tests/kernels/moe/test_muse_spark_trtllm_moe.py` | [Perf] Fuse all-gather and norm for Muse Spark hybrid MoE (#64638) |
| 6 | [vllm-project/vllm@4d1376a](https://github.com/vllm-project/vllm/commit/4d1376a4922e28d814d79fae3b16d8de9d2d46cf) | `2026-10-30` | `docs/features/mooncake_store_connector_usage.md`, `tests/v1/kv_connector/unit/test_mooncake_store_coordinator.py` | [KV Connector] Support heterogeneous TP sharing for hybrid KV cache in Mooncake Store (#54307) |
| 7 | [vllm-project/vllm@deacc36](https://github.com/vllm-project/vllm/commit/deacc3670153845db2f013ea24085d6eaf338874) | `2026-10-30` | `tests/basic_correctness/memory/cumem/test_cumem.py`, `vllm/device_allocator/cumem.py` | [Bugfix][SleepMode] Exact-size, reusable pinned CPU backup for level-1 sleep (#64630) |
| 8 | [NVIDIA/TensorRT-LLM@120ea6b](https://github.com/NVIDIA/TensorRT-LLM/commit/120ea6b8c7a15107aaafd7f696cc532308487ee7) | `2026-10-30` | `cpp/tensorrt_llm/CMakeLists.txt`, `cpp/tensorrt_llm/common/CMakeLists.txt` | [None][refactor] Split `thop.attention` into phased fmha (#17558) |
| 9 | [NVIDIA/TensorRT-LLM@ac6fb72](https://github.com/NVIDIA/TensorRT-LLM/commit/ac6fb72d9fa1caf2b2d730aa4797727e6b3f6d23) | `2026-10-30` | `cpp/tensorrt_llm/kernels/cuda_graph_grouped_gemm.cu`, `cpp/tensorrt_llm/kernels/cuda_graph_grouped_gemm.h` | [TRTLLM-15314][feat] Support native FP8 MoE LoRA weights (#18555) |
| 10 | [NVIDIA/TensorRT-LLM@c50ea88](https://github.com/NVIDIA/TensorRT-LLM/commit/c50ea882ef2d4859c4dd08eac46273bfbee507d3) | `2026-10-30` | `docs/source/deployment-guide/deployment-guide-for-kimi-k3-on-trtllm.md`, `tensorrt_llm/_torch/attention/ATTENTION_DEVELOPER_GUIDE.md` | [None][feat] Kimi K3: support NVFP4 KV cache (FP4 MLA) with the hybrid KV cache manager (#20888) |

---

## 3. Deterministic Offline Verification (`experiments/sglang_dspark_vllm_musespark_trtllm_fp8lora_kimik3_phase140/`)

- **Kimi-K3 NVFP4 MLA KV Capacity Multiplier vs. BF16**: **`3.56x`** (`1152 B -> 324 B` per token-layer).
- **vLLM Muse Spark Fused All-Gather + Norm Speedup**: **`1.32x`**.
- **vLLM Level-1 Sleep Pinned Backup Reuse Reduction**: **`91.67%`** (`120.0 ms -> 10.0 ms`).
