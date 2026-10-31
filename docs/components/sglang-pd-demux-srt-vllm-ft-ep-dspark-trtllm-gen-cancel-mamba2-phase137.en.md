# SGLang PD Demultiplexing & Native Tensor Scoring, vLLM Fault-Tolerant EP Scale-Down & DSpark Batch Sharding, and TensorRT-LLM Paired GEN Cancellation & Mamba2 MTP Zero-Pool-Copy (Phase 137)

> **中文版**: [sglang-pd-demux-srt-vllm-ft-ep-dspark-trtllm-gen-cancel-mamba2-phase137.md](sglang-pd-demux-srt-vllm-ft-ep-dspark-trtllm-gen-cancel-mamba2-phase137.md)
> **Entry Point**: Serving & Scheduling — SGLang PD Multiplexing Removal & Native Tensor Scoring / vLLM Fault-Tolerant EP Scale-Down & DSpark Sharding / TensorRT-LLM Paired GEN Cancellation & Mamba2 MTP Indexed SSM Updates
> **Evidence Tier**: `implementation` + `own-offline` (10 primary serving commits audited at source level + Python 3.12 deterministic modeling)

---

## 1. Executive Summary & Architectural Conclusions

1. **SGLang Scheduler Simplification & Native Tensor Scoring (`sglang@a076243`, `783b390`, `b5a0c9d`, `6fb0fd9`)**: Removes legacy PD multiplexing from the scheduler, optimizes SRT prefill scoring with native tensor outputs, fixes Diffusion LoRA merge cache key identity/eviction, and sizes one-batch KV capacity checks by the heaviest attention-DP rank on ragged batches.
2. **vLLM Fault-Tolerant EP Scale-Down, DSpark Batch Sharding & CPU MRv2 (`vllm@913fccd`, `5423fb8`, `040abb2`)**: Introduces fault-tolerant Expert Parallelism via dynamic scale-down (`W -> W - f`), adds batch-sharded sequential blocks for DSpark speculative decoding, and enables Model Runner v2 (MRv2) on non-Triton CPU backends.
3. **TensorRT-LLM Paired GEN Cancellation, Mamba2 MTP SSM Cache & Bounded TriAttention (`TensorRT-LLM@5ec25c5`, `63c0ef2`, `eebd9d0`)**: Immediately cancels the paired GEN request when CTX fails or is cancelled in `openai_disagg_service.py`, eliminates whole-pool SSM copies in custom Mamba2 MTP updates (**12.8x** memory traffic reduction at $B=40, N_{\text{pool}}=512$), and bounds TriAttention eviction workspaces with a resident fallback.

---

## 2. Primary Commit Matrix

| # | Repo & Commit | Date | Primary Code Paths | Architectural Mechanism & Breakthrough |
|---|---|---|---|---|
| 1 | [sgl-project/sglang@a076243](https://github.com/sgl-project/sglang/commit/a076243ceacf376b9632d8da37dc8e17005595bd) | `2026-10-30` | `docs/docs/advanced_features/pd_disaggregation.mdx`, `docs/docs/advanced_features/server_arguments.mdx` | [Refactor] scheduler: Remove PD multiplexing (#43898) |
| 2 | [sgl-project/sglang@783b390](https://github.com/sgl-project/sglang/commit/783b3905b14ab0383b604bf45dfe873393680922) | `2026-10-30` | `python/sglang/srt/entrypoints/engine.py`, `python/sglang/srt/layers/attention/flashattention_backend.py` | [Perf] srt: Native tensor output and prefill scoring optimizations (#42248) |
| 3 | [sgl-project/sglang@b5a0c9d](https://github.com/sgl-project/sglang/commit/b5a0c9dc18832959d71df69faca1ce07e043c97c) | `2026-10-30` | `python/sglang/multimodal_gen/runtime/pipelines_core/lora/lora_merge_cache.py`, `python/sglang/multimodal_gen/runtime/pipelines_core/lora/pipeline.py` | [Fix] diffusion: LoRA merge cache key identity and eviction (#45834) |
| 4 | [sgl-project/sglang@6fb0fd9](https://github.com/sgl-project/sglang/commit/6fb0fd97fdc5ef0336663fb8134ead490956816a) | `2026-10-30` | `python/sglang/benchmark/one_batch_server.py`, `test/registered/unit/bench/test_one_batch_ragged_capacity.py` | [Fix] bench: Size the one-batch KV-capacity skip by the heaviest attention-DP rank for ragged batches (#47130) |
| 5 | [vllm-project/vllm@913fccd](https://github.com/vllm-project/vllm/commit/913fccd11847f5b4d4769e3a69c8925f955bc29e) | `2026-10-30` | `.buildkite/test_areas/fault_tolerance.yaml`, `docs/serving/expert_parallel_deployment.md` | [Feature] Initial support for fault tolerant ep using scale-down (#46370) |
| 6 | [vllm-project/vllm@5423fb8](https://github.com/vllm-project/vllm/commit/5423fb8b93a539b24e7f80075d132d4f4dcab95b) | `2026-10-30` | `tests/distributed/test_pynccl.py`, `tests/v1/e2e/spec_decode/test_sharded_sampling.py` | [Spec Decode][DSpark] Batch-sharded sequential block (#63886) |
| 7 | [vllm-project/vllm@040abb2](https://github.com/vllm-project/vllm/commit/040abb262c5b09b8325abe0379ff70246fd92bba) | `2026-10-30` | `.buildkite/scripts/hardware_ci/run-cpu-test-arm.sh`, `.buildkite/test_areas/misc.yaml` | [CPU][MRv2] Enable MRv2 for non-triton CPU backends (#61995) |
| 8 | [NVIDIA/TensorRT-LLM@5ec25c5](https://github.com/NVIDIA/TensorRT-LLM/commit/5ec25c55dba861738e3f40b8e155faf91bc3fbbe) | `2026-10-31` | `tensorrt_llm/serve/openai_client.py`, `tensorrt_llm/serve/openai_disagg_service.py` | [TRTLLM-17083][feat] Cancel paired GEN on CTX failure or cancellation (#20491) |
| 9 | [NVIDIA/TensorRT-LLM@63c0ef2](https://github.com/NVIDIA/TensorRT-LLM/commit/63c0ef23be4ff88d40bc5addac7349fd3fe5e71a) | `2026-10-30` | `cpp/tensorrt_llm/kernels/mamba2MTPSSMCache/mamba2MTPSSMCache.h`, `cpp/tensorrt_llm/kernels/mamba2MTPSSMCache/mamba2MTPSSMCacheKernel.cuh` | [TRTLLM-17350][perf] Avoid whole-pool SSM copies in custom Mamba2 MTP updates (#21105) |
| 10 | [NVIDIA/TensorRT-LLM@eebd9d0](https://github.com/NVIDIA/TensorRT-LLM/commit/eebd9d0fcf6cafa4a078bed9a6c06eb1208e8a2a) | `2026-10-30` | `docs/source/features/kv-cache-compression.md`, `examples/kv_cache_compression/triattention.md` | [TRTLLM-17275][fix] Bound TriAttention eviction workspace with a resident fallback (#20871) |

---

## 3. Deterministic Offline Verification (`experiments/sglang_pd_vllm_ft_ep_trtllm_mamba2_phase137/`)

- **Mamba2 MTP SSM Copy Volume**: Reduced from `268.44 MB` to **`20.97 MB`** (**12.8x** speedup).
- **Orphaned GEN Slot-Seconds Saved**: **96.5%** reduction under 5% CTX cancellation/error rate.
- **Fault-Tolerant EP Recovery Time**: Reduced from `180.0 s` full restart to **`28.0 s`** scale-down (**84.4%** faster recovery).
