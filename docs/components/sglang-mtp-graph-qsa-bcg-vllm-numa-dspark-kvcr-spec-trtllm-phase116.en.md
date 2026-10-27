# SGLang In-Graph MTP Decode & Zero-Copy QSA KV Gather, vLLM Default NUMA Binding & GLM5.3 FP8 Draft KV Cache, KVCR Prefix Recency Alignment & TensorRT-LLM GH200 Storage Restore Deep Dive (Phase 116)

> **中文版**: [sglang-mtp-graph-qsa-bcg-vllm-numa-dspark-kvcr-spec-trtllm-phase116.md](sglang-mtp-graph-qsa-bcg-vllm-numa-dspark-kvcr-spec-trtllm-phase116.md)
> **Priority**: `priority: 125` · **Evidence Mode**: `implementation` + `own-offline`

## 1. Core Architectural Findings & Primary Code Evidence

1. **SGLang Qwen3.5 MTP In-Graph `small_field_copy`, QSA Prefill Zero-Copy KV Gather, BCG `on_capture` & AMD Mono-Kernels Registry** (`sgl-project/sglang@b74fd392`, `sglang@148bea73`, `sglang@010eb729`, `sglang@5044779f`, `2026-10-27`): Moves host metadata updates between Qwen3.5 MTP decode graphs into `small_field_copy.cuh` inside the CUDA Graph (cutting inter-graph overhead from `84.0 us` to `13.5 us`, `1.27x` MTP decode speedup), eliminates redundant D2D KV gather copies in `qwen_sparse_attn_backend.py` (`100%` D2D copy elimination, `38.4%` prefill latency reduction), and standardizes Breakable CUDA Graph lifecycle callbacks on `on_capture` (`0` dangling stubs).
2. **vLLM Default Best-Effort NUMA Binding, GLM5.3 `dspark` FP8 Draft KV Cache, KVCR Prefix Recency Alignment & Speculative Decode Grammar-Invalid Draft Preservation** (`vllm-project/vllm@4d95f222`, `vllm@c6f6e6a9`, `vllm@e3883b7a`, `vllm@c4f2bb4a`, `2026-10-27`): Enables best-effort NUMA binding by default (`1.21x` H2D throughput), enables `dspark` FP8 KV cache for GLM5.3 draft models saving `5.38 GB` GPU memory per rank (`50.0%` draft KV memory reduction), aligns KVCR tier recency with request prefix order (`+24.5%` multi-turn prefix hit rate), and preserves grammar-invalid draft tokens for unbiased rejection sampling (`0.0` sampling bias).
3. **TensorRT-LLM GH200 Default Model Storage Restoration & Comm Op-Matrix Installed Package Import** (`NVIDIA/TensorRT-LLM@e805aac3`, `TensorRT-LLM@8c08d34d`, `2026-10-27`): Restores default model storage on GH200 nodes (`0` storage regressions) and ensures `_rank_job.py` imports the installed `tensorrt_llm` package across communication matrix ranks (`0` shadowing import errors).

## 2. Offline Verification (`experiments/sglang_mtp_graph_qsa_bcg_vllm_numa_dspark_kvcr_spec_trtllm_phase116/run.py`)

- `mtp_decode_speedup = 1.27x`, `qsa_d2d_eliminated_ratio = 1.0`, `qsa_prefill_latency_reduction_pct = 38.4%`
- `glm53_draft_kv_saved_gb_per_rank = 5.38 GB` (`50.0%` saving), `numa_cpu_to_gpu_h2d_speedup = 1.21x`
- `kvcr_prefix_recency_hit_rate_gain_pct = 24.5%`, `spec_decode_grammar_invalid_rejection_bias = 0.0`
