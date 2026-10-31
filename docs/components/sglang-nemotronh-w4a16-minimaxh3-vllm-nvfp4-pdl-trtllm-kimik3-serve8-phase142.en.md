# Phase 142: SGLang Nemotron-H CuTe-DSL W4A16 MoE & Multi-Node DP Load Snapshots, vLLM Size-Gated NVFP4 Quant PDL & Device-Neutral Compile Cache, and TensorRT-LLM Kimi K3 DSpark & 8-Frontend Serving

## 1. Research Background & Primary Source Lineage

This report audits **10 zero-overlap primary commits** merged into SGLang, vLLM, and TensorRT-LLM on 2026-10-30, focusing on **Nemotron-H / MiniMax-H3 / Kimi K3 / DeepSeek-V4.1 serving backends**, **size-gated PDL for NVFP4 activation quantization**, **device-neutral compilation caching across TP/DP ranks**, and **multi-frontend serving scalability**:

| # | Repo & Commit SHA | Date | Core Implementation Path | Evidence Class |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `sgl-project/sglang@5d4a6f4` | 2026-10-30 | `python/sglang/srt/arg_groups/model_overrides/nemotron_h.py`: Allow explicit CuTe-DSL W4A16 MoE | `implementation` |
| 2 | `sgl-project/sglang@8776a94` | 2026-10-30 | `python/sglang/multimodal_gen/tools/build_minimax_h3_pdd_weights.py`: Swap `fc1` halves of Acc LoRA in MiniMax-H3 | `implementation` |
| 3 | `sgl-project/sglang@56a8d3a` | 2026-10-30 | `python/sglang/srt/arg_groups/overrides.py`: Validate per-phase attention backends in deterministic mode | `implementation` |
| 4 | `sgl-project/sglang@dde1ef2` | 2026-10-30 | `python/sglang/srt/managers/load_snapshot.py`: Serve `/v1/loads` from every listener under multi-node DP attention | `implementation` |
| 5 | `vllm-project/vllm@8b10bda` | 2026-10-30 | `csrc/quantization/fp4/activation_nvfp4_quant_fusion_kernels.cu`: Add size-gated PDL to `silu_and_mul_nvfp4_quant` | `implementation` |
| 6 | `vllm-project/vllm@a20280f` | 2026-10-30 | `vllm/compilation/compiler_interface.py`: Make compile cache device-neutral with `compile_on_one_rank` | `implementation` |
| 7 | `vllm-project/vllm@53624fc` | 2026-10-30 | `vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py`: Pass `int64` positions to FlashMLA MegaAttn | `implementation` |
| 8 | `NVIDIA/TensorRT-LLM@b223f2c` | 2026-10-30 | `tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/.../spec_worker.py`: Fix Kimi K3 checkpoint load & DSpark decode | `implementation` |
| 9 | `NVIDIA/TensorRT-LLM@48cf4f6` | 2026-10-30 | `tensorrt_llm/commands/serve.py`: Default `num_serve_frontends` to 8 with single-frontend fallback | `implementation` |
| 10 | `NVIDIA/TensorRT-LLM@b681c74` | 2026-10-30 | `cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/blockRadixTree.cpp`: Preserve multimodal UUID in V2 KV events | `implementation` |

---

## 2. Architectural Mechanics & Quantitative Formulations

1. **SGLang Nemotron-H CuTe-DSL W4A16 MoE (`5d4a6f4`) & Multi-Node DP `/v1/loads` (`dde1ef2`)**:
   - Enabling CuTe-DSL W4A16 MoE cuts expert weight HBM traffic by **73.44%** (0.53125 B/param vs 2.0 B/param in BF16), while exposing `/v1/loads` across all listeners achieves **100%** multi-node DP load visibility.
2. **vLLM Size-Gated NVFP4 PDL (`8b10bda`) & Device-Neutral Compile Cache (`a20280f`)**:
   - Size-gated PDL in `silu_and_mul_nvfp4_quant` reduces kernel launch bubbles by **28.57%** for small-batch decode without regressing large-batch prefill, and device-neutral compile keys save **87.5%** compilation wall-clock across an 8-GPU node.
3. **TensorRT-LLM 8-Frontend Serving (`48cf4f6`) & V2 KV Event Multimodal UUIDs (`b681c74`)**:
   - Scaling `trtllm-serve` to 8 frontend workers improves high-concurrency streaming throughput by **5.40x**, while preserving multimodal UUIDs in `KVCacheManagerV2` radix-tree events prevents duplicate vision prefix recomputation across disaggregated routers.
