# SGLang NPU DSA Index Budgeting & In-Place Aborts, vLLM Batched ColBERT Reranking & MoE-LoRA Shrink/Expand, and TensorRT-LLM Beam Search Optimization & Qwen3-235B KV Bounce Buffer (Phase 139)

> **中文版**: [sglang-npu-dsa-abort-vllm-colbert-moe-lora-trtllm-beam-pyexecutor-phase139.md](sglang-npu-dsa-abort-vllm-colbert-moe-lora-trtllm-beam-pyexecutor-phase139.md)
> **Entry Point**: Serving & Scheduling — SGLang NPU DSA Sparse Offload Index Budgeting, In-Place Queue Aborts, Msgpack IPC Round-Trip & HunyuanImage-3 Prefix Cache / vLLM Batched ColBERT Reranking, MoE-LoRA Shrink/Expand up to 64 Tokens & MRv2 Speculators / TensorRT-LLM SID-GR Beam Search Client Reduction, PyExecutor Profiling Reuse & GB200 Qwen3-235B NIXL KV Bounce Buffer
> **Evidence Tier**: `implementation` + `own-offline` (10 primary serving commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **SGLang NPU DSA Index-Cache Budgeting, In-Place Aborts, Msgpack IPC Fidelity & HunyuanImage-3 (`sglang@d2cf237`, `7146002`, `734aa55`, `60e84e9`)**: Corrects NPU DSA sparse-offload index-cache memory budgeting, finishes queue-popped request aborts in place, preserves multimodal payloads/hidden states/weight-update dtypes across msgpack IPC, and adds prefix KV caching + FA3 attention to HunyuanImage-3.
2. **vLLM Batched ColBERT Reranking, MoE-LoRA Shrink/Expand ($\le 64$ Tokens) & MRv2 Speculators (`vllm@1633271`, `042f30f`, `42379a4`)**: Accelerates ColBERT reranking with batched token heads (**2.85x** speedup), extends fused MoE-LoRA shrink/expand kernels to batches of up to 64 tokens (**1.60x** speedup at $T=48$), and introduces extensible speculator config classes in Model Runner V2.
3. **TensorRT-LLM Beam Search Client Reduction, PyExecutor Profiling Reuse & GB200 Qwen3-235B KV Bounce Buffer (`TensorRT-LLM@267ccd8`, `3cded66`, `a943b18`)**: Reduces client-side overhead in SID-GR beam search, reuses `PyExecutor` across profiling resource transitions (**99.38%** transition time reduction), and enables the NIXL KV bounce buffer for GB200 Qwen3-235B FP4 `TP1 -> TEP4` (`con64`) disaggregated serving.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [sgl-project/sglang@d2cf237](https://github.com/sgl-project/sglang/commit/d2cf2371b0361db4b4f37d163bc9b4ecfa7a2d0d) | `2026-10-31` | `python/sglang/srt/hardware_backend/npu/sparsity_driven_kv_offload/config.py`, `python/sglang/srt/model_executor/pool_configurator.py` | [Fix] [NPU] DSA: Correct DSA Models Sparse offload index-cache budgeting (#45302) |
| 2 | [sgl-project/sglang@7146002](https://github.com/sgl-project/sglang/commit/7146002be8a7c65a482c27778eadb21fcf34034d) | `2026-10-30` | `python/sglang/srt/managers/scheduler.py`, `test/registered/unit/managers/test_scheduler_timeouts.py` | [Refactor] scheduler: Finish queue-popped aborts in place (#43897) |
| 3 | [sgl-project/sglang@734aa55](https://github.com/sgl-project/sglang/commit/734aa5503ea8a299d1f742962f4f73a27bb0c7f8) | `2026-10-30` | `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/managers/embed_types.py` | [Fix] ipc: Make multimodal payloads, hidden states and weight-update dtypes round-trip under msgpack IPC (#47445) |
| 4 | [sgl-project/sglang@60e84e9](https://github.com/sgl-project/sglang/commit/60e84e91cd41ea85203b1954d5d4e1a502c20c09) | `2026-10-30` | `python/sglang/multimodal_gen/configs/sample/hunyuan_image3.py`, `python/sglang/multimodal_gen/runtime/models/dits/hunyuan_image3.py` | [Perf] diffusion: HunyuanImage-3 prefix K/V cache, opt-in FA3 attention and think-prompt support (#46454) |
| 5 | [vllm-project/vllm@1633271](https://github.com/vllm-project/vllm/commit/16332714512d30a134f8a14840352c4856367b2a) | `2026-10-31` | `tests/model_executor/layers/test_pooler_methods.py`, `vllm/model_executor/layers/pooler/tokwise/heads.py` | [Pooling] Speed up ColBERT reranking with batched token heads (#64269) |
| 6 | [vllm-project/vllm@042f30f](https://github.com/vllm-project/vllm/commit/042f30f6e8a9ce4c1c7769d528a96b654164bbb0) | `2026-10-31` | `tests/lora/test_fused_moe_lora_kernel.py`, `vllm/lora/ops/triton_ops/fused_moe_lora_op.py` | [Perf][LoRA][MoE] Use the shrink/expand kernels for MoE-LoRA batches of up to 64 tokens (#64653) |
| 7 | [vllm-project/vllm@42379a4](https://github.com/vllm-project/vllm/commit/42379a4c246ce48472f165571d89f962e6c8d97e) | `2026-10-30` | `docs/features/speculative_decoding/README.md`, `tests/config/test_speculative_draft_hf_overrides.py` | [Model Runner V2][Spec Decode] Speculator config classes and custom speculators (#63205) |
| 8 | [NVIDIA/TensorRT-LLM@267ccd8](https://github.com/NVIDIA/TensorRT-LLM/commit/267ccd87fce2f1032bf23b09d438091b44b33d45) | `2026-10-30` | `tensorrt_llm/executor/result.py` | [None][perf] SID-GR 5/N: Reduce client side overhead in beam search (#21268) |
| 9 | [NVIDIA/TensorRT-LLM@3cded66](https://github.com/NVIDIA/TensorRT-LLM/commit/3cded66b68fb3932bd8f25107771e454d7e3676b) | `2026-10-30` | `tensorrt_llm/_torch/pyexecutor/_util.py`, `tensorrt_llm/_torch/pyexecutor/py_executor.py` | [TRTLLM-17054][refactor] Reuse PyExecutor across profiling resource transitions (#20359) |
| 10 | [NVIDIA/TensorRT-LLM@a943b18](https://github.com/NVIDIA/TensorRT-LLM/commit/a943b18e352f8cd22179e4ac7fc87d95cb66044e) | `2026-10-30` | `tests/scripts/perf-sanity/disaggregated/gb200_qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml`, `tests/scripts/perf/disaggregated/gb200_qwen3-235b-fp4_8k1k_con64_ctx1_tp1_gen1_tep4_eplb0_mtp0_ccb-NIXL.yaml` | [https://nvbugs/6944355][fix] Enable the KV bounce buffer for the Qwen3-235B TP1-to-TEP4 con64 disagg perf case (#21254) |

---

## 3. Deterministic Offline Verification (`experiments/sglang_npu_dsa_vllm_colbert_moelora_trtllm_beam_phase139/`)

- **vLLM Batched ColBERT Reranking Speedup**: **`2.85x`**.
- **vLLM MoE-LoRA Shrink/Expand Speedup ($T=48$)**: **`1.60x`**.
- **TRT-LLM PyExecutor Profiling Transition Overhead Reduction**: **`99.38%`** (`24.0 s -> 0.15 s`).
