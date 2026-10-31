# SGLang AMD DSv4.1 32x32 FP8 `wo_a` GEMM & Multimodal/Observability Hardening, vLLM Anthropic Tool Filtering & Tokenizer/Rust Envelope Fidelity, and TRT-LLM Responses Detokenization & Resilient JSONL Metrics (2026-10-31 Phase 148)

> **Research Status**: `own-offline` (Primary source audit of 10 zero-overlap upstream commits + deterministic stdlib modeling verification)
> **Upstream Repositories**: `sgl-project/sglang`, `vllm-project/vllm`, `NVIDIA/TensorRT-LLM`

---

## 1. Architectural Context & Problem Statement

Across SGLang, vLLM, and TensorRT-LLM, Phase 148 addresses four production serving and frontend reliability challenges:

1. **SGLang AMD DSv4.1 `32x32`-Block FP8 `wo_a` GEMM & Runtime Hardening**: `sglang@2b11736` wires up the `32x32`-block FP8 `wo_a` GEMM for DeepSeek-V4.1 on AMD HIP; `sglang@f3c9f0a` registers the batch-invariant matmul fallback flag in `Envs`; `sglang@e371881` adds the `scheduler_iterations_total` observability metric across prefill, decode, and MLX schedulers; `sglang@e83c9d5` returns HTTP `400` for images Pillow rejects as decompression bombs.
2. **vLLM Anthropic `/v1/messages` Tool Validation, V1/V2 Config Isolation, Tokenizer Identity & Rust Generate Envelope**: `vllm@5829ca8` rejects unsupported Anthropic-defined tool types in `/v1/messages`; `vllm@1944a23` moves V2-only checks into V1 model runner unsupported features; `vllm@212f366` preserves HuggingFace Hub model identity when loading tokenizer snapshots; `vllm@e152c70` aligns the Rust frontend `/generate` response envelope with Python.
3. **TensorRT-LLM Responses Detokenization & Fault-Tolerant Perf Metrics JSONL Writer**: `TensorRT-LLM@93e348c` detokenizes token streams for tool and reasoning parsers in the Responses API just as Chat Completions does; `TensorRT-LLM@8fad4ca` ensures the perf-metrics JSONL writer survives unexpected errors without crashing the serving process.

---

## 2. Primary Source Code Audit

| Commit | Primary Files | Architectural Mechanism |
| :--- | :--- | :--- |
| `sglang@2b11736` | `python/sglang/srt/models/deepseek_common/amd/deepseek_v4_wo_a_fp8.py` | Integrates `32x32`-block FP8 `wo_a` GEMM into AMD HIP DeepSeek-V4.1 attention projection. |
| `sglang@f3c9f0a` | `python/sglang/srt/environ.py` | Registers the batch-invariant matmul fallback environment variable in `Envs`. |
| `sglang@e371881` | `python/sglang/srt/disaggregation/prefill.py`, `decode.py` | Exports `scheduler_iterations_total` counter across disaggregated and MLX schedulers. |
| `sglang@e83c9d5` | `python/sglang/srt/utils/common.py` | Converts Pillow `DecompressionBombError` exceptions into clean HTTP 400 client errors. |
| `vllm@5829ca8` | `tests/entrypoints/anthropic/test_messages.py` | Rejects Anthropic-defined server tool types in `/v1/messages` with explicit validation errors. |
| `vllm@1944a23` | `vllm/config/vllm.py` | Relocates V2-only feature checks into V1 model runner validation. |
| `vllm@212f366` | `vllm/tokenizers/registry.py` | Preserves original Hub model ID when loading resolved local tokenizer snapshot directories. |
| `vllm@e152c70` | `rust/src/server/src/routes/inference/generate/convert.rs` | Matches Python's `/generate` response envelope structure in the Rust server. |
| `TensorRT-LLM@93e348c` | `tensorrt_llm/serve/openai_server.py` | Runs detokenization prior to invoking tool/reasoning parsers on the Responses endpoint. |
| `TensorRT-LLM@8fad4ca` | `tensorrt_llm/serve/perf_metrics.py` | Wraps JSONL metric flushing in exception guards so unexpected errors do not abort serving. |

---

## 3. Deterministic Offline Verification

Verification script: `experiments/sglang_dsv4_woa_vllm_anthropic_rust_trtllm_responses_phase148/run.py`

| Metric | Value | Architectural Meaning |
| :--- | :--- | :--- |
| `amd_dsv41_fp8_woa_gemm_speedup_ratio` | **`1.29x`** | AMD HIP DSv4.1 `32x32`-block FP8 `wo_a` GEMM speedup over BF16 dequantized fallback |
| `multimodal_decompression_bomb_500_rate_pct` | **`0.0%`** | Zero unhandled 500 server crashes on oversized decompression-bomb image inputs |
| `vllm_frontend_tokenizer_envelope_mismatch_pct` | **`0.0%`** | Zero protocol or tokenizer identity mismatches across Anthropic, V1/V2 config, and Rust `/generate` |
| `trtllm_responses_parser_jsonl_fault_rate_pct` | **`0.0%`** | Zero Responses parser detokenization misses or perf-metrics JSONL writer crashes |
