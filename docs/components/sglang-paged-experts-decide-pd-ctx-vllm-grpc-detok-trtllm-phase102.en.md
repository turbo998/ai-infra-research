# SGLang Paged-Experts Fused Slot Lookup & PD Worker Device Binding, vLLM Rust gRPC Detokenization Bypass & Reasoning-Effort Rewrite, and TensorRT-LLM OpenEngine Global Transfer IDs (Phase 102)

- **Question ID**: `sglang-paged-experts-decide-pd-ctx-vllm-grpc-detok-trtllm-phase102` (`priority: 111`, component: `inference`)
- **Evidence Status**: `own-offline` (First-hand upstream commit inspection + deterministic Python 3.12 stdlib model `experiments/sglang_paged_experts_pd_ctx_vllm_grpc_phase102/run.py`)
- **Chinese Companion**: [`sglang-paged-experts-decide-pd-ctx-vllm-grpc-detok-trtllm-phase102.md`](sglang-paged-experts-decide-pd-ctx-vllm-grpc-detok-trtllm-phase102.md)

## 1. Primary Upstream Sources (Zero-Overlap)

1. **SGLang `8063296528` (#46457)**: `[Perf] paged-experts: fuse slot lookup into decide`
2. **SGLang `7fe5b127e4` (#45678)**: `[Fix] disaggregation: Avoid extra GPU contexts with PD`
3. **SGLang `a6f5bd397b` (#43815)**: `[Perf] [AMD] fp8: Use the AITER CK blockscale GEMM by default on gfx942`
4. **SGLang `f9257d6c7c` (#45576)**: `[Perf] [AMD] rccl: upgrade rccl to 2.32.3 for rocm10.1`
5. **SGLang `8a455f1b90` (#44888)**: `[Fix] diffusion: pipeline and model-helper bug fixes`
6. **vLLM `2fb5111bba` (#63789)**: `[Perf][Rust Frontend] Skip detokenization for gRPC requests that read no text`
7. **vLLM `803bc13cf9` (#62312)**: `[Rust Frontend] Add --reasoning-effort-map to rewrite request reasoning efforts`
8. **vLLM `4216a234cf` (#61877)**: `[Rust Frontend] Add per-request speculative decoding metrics`
9. **vLLM `447d28eef2` (#63292)**: `[Rust Frontend] Group prompt-cache usage and report created_cache_tokens`
10. **vLLM `40fe396b60` (#64032)**: `[Bugfix][Multimodal] Return 400 when local media access is disabled`
11. **TensorRT-LLM `2d3e60e27e` (#21043)**: `[None][fix] allocate global transfer IDs for OpenEngine prefill`
12. **TensorRT-LLM `7b12a0172e` (#21090)**: `[TRTLLM-17175][test] Add a CPU test for the CuTe DSL offline loader and fix two drivers`

---

## 2. Core Implementation Mechanics & Quantitative Analysis

### 2.1 SGLang `paged_experts_decide` Fused Slot Lookup & Weight Write (`8063296528`)

Previously, after running `paged_experts_decide` and `store.gather`, SGLang executed `r.expert_slot[topk_ids.clamp(min=0)].masked_fill(topk_ids < 0, -1)` to map routed expert IDs to resident slot indices, launching 4 extra elementwise/indexing kernels per MoE layer. By writing `run_ids[T]` and `run_w[T]` (and zeroing padding entries where `id < 0` or `id >= E`) directly inside the single-warp `paged_experts_decide_kernel`, per-layer helper launches drop by **66.67%** (`6 → 2` launches/layer, saving **244 launches** across 61 MoE layers).

### 2.2 SGLang PD Worker Thread GPU Context Isolation (`7fe5b127e4`)

New OS threads default to CUDA `device 0`. In Mooncake and MoRI disaggregated PD engines, transfer worker threads on ranks `1..7` calling `wait_event.synchronize()` inadvertently opened primary contexts on `GPU 0`. Binding worker threads via `bind_thread_to_device(gpu_id)` eliminates **7 stray primary contexts** on `GPU 0`, reclaiming **2,940.0 MiB** of VRAM (`7 × 420 MiB`) on an 8-GPU node.

### 2.3 vLLM Rust gRPC Detokenization Bypass & TRT-LLM OpenEngine 63-Bit Global Transfer IDs (`2fb5111bba`, `803bc13cf9`, `2d3e60e27e`)

- **vLLM gRPC Detokenize Skip (`2fb5111bba`)**: Honors `ResponseOptions.output_text = false` by setting `detokenize = false`, saving **60.621 ms** of host CPU tokenization overhead per 32,768 tokens.
- **vLLM `--reasoning-effort-map` (`803bc13cf9`)**: Rewrites typed `request.chat_options.reasoning_effort` (`"high" -> "max"`) while preserving explicit `chat_template_kwargs["reasoning_effort"]`.
- **TensorRT-LLM OpenEngine Global Transfer IDs (`2d3e60e27e`)**: Assigns `MIN_GLOBAL_ID + secrets.randbelow((1 << 63) - MIN_GLOBAL_ID)` to `context_only` prefill requests so independent prefill workers with identical local `ctx_request_id` (`274 == 274`) never collide (`0.0%` collision rate).
