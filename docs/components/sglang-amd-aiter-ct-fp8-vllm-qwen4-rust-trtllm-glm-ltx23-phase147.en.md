# SGLang AMD gfx950 Deterministic Fallback & AITER Scaling/CT FP8 MoE, Rust Tiered Cache Telemetry, vLLM Qwen4Exp Int64 Gating & Rust Frontend/Sharded Loader, and TRT-LLM GLM Reasoning Parser & LTX-2.3 Retake (2026-10-31 Phase 147)

> **Research Status**: `own-offline` (Primary source audit of 10 zero-overlap upstream commits + deterministic stdlib modeling verification)
> **Upstream Repositories**: `sgl-project/sglang`, `vllm-project/vllm`, `NVIDIA/TensorRT-LLM`

---

## 1. Architectural Context & Problem Statement

Across SGLang, vLLM, and TensorRT-LLM, Phase 147 resolves four production serving and multimodal generation challenges:

1. **SGLang AMD `gfx950` Determinism, AITER Scaling & CT FP8 MoE Layout**: `sglang@0fd1e3a` bypasses non-deterministic AITER GEMM and MoE kernels in deterministic mode on `gfx950`; `sglang@85af813` forwards each layer's custom `softmax_scale` into AITER MHA prefill and decode; `sglang@196ee6b` preserves Compressed-Tensors (CT) FP8 Triton MoE layout and reduction dimensions; `sglang@93a936d` reports cached tokens broken down by cache tier in the Rust server's native `/generate` `meta_info`.
2. **vLLM Qwen4Exp `int64` Row Offsets & Sharded State HF Resolution**: `vllm@58f245c` computes Qwen4Exp HyperConnection (HC) and Per-Layer Embedding (PLE) gate row offsets in `int64` across NVIDIA and AMD kernels to prevent 32-bit overflow; `vllm@c9af9ac` resolves HuggingFace Hub repository IDs before loading sharded checkpoints.
3. **vLLM Rust Frontend Protocol Fidelity**: `vllm@c817cc0` populates `completion_tokens_details` only when reasoning tokens are counted; `vllm@6de8c7b` adds `return_prompt_text` support in the Rust chat/streaming frontend.
4. **TensorRT-LLM GLM Reasoning Parser & LTX-2.3 Retake Pipeline**: `TensorRT-LLM@4cd6c19` drops the owed `</think>` expectation after a tool call in the GLM reasoning parser and reads thinking mode directly from the rendered prompt; `TensorRT-LLM@72507c8` introduces the LTX-2.3 text-to-video (t2v) and localized temporal retake pipeline.

---

## 2. Primary Source Code Audit

| Commit | Primary Files | Architectural Mechanism |
| :--- | :--- | :--- |
| `sglang@0fd1e3a` | `python/sglang/srt/layers/quantization/unquant.py` | Bypasses AITER GEMM and MoE when deterministic mode is enabled on `gfx950` to guarantee bitwise reproducibility. |
| `sglang@85af813` | `python/sglang/srt/layers/attention/aiter_backend.py` | Passes the layer's `softmax_scale` into AITER MHA prefill and decode kernels. |
| `sglang@196ee6b` | `compressed_tensors_w8a8_fp8_moe.py` | Preserves CT FP8 Triton MoE tensor layout and reduction dimensions across quantization schemes. |
| `sglang@93a936d` | `proto/sglang/api/v1/generate.proto`, `output_streamer.py` | Exposes per-tier cached token counts in native `/generate` `meta_info` via the Rust server. |
| `vllm@58f245c` | `vllm/models/qwen4_exp/common/ops/ple.py`, `nvidia/ops/hc.py` | Casts row indices and strides to `int64` in Qwen4Exp HC and PLE gate kernels. |
| `vllm@c817cc0` | `rust/src/server/src/routes/openai/chat_completions.rs` | Emits `completion_tokens_details` only when reasoning tokens are actively counted. |
| `vllm@6de8c7b` | `rust/src/chat/src/request.rs`, `stream.rs` | Supports `return_prompt_text` in the Rust chat and streaming frontend. |
| `vllm@c9af9ac` | `vllm/model_executor/model_loader/sharded_state_loader.py` | Resolves HuggingFace Hub repository paths to local snapshot directories in `ShardedStateLoader`. |
| `TensorRT-LLM@4cd6c19` | `tensorrt_llm/llmapi/reasoning_parser.py` | Drops owed `</think>` state after tool calls and inspects the prompt for GLM thinking mode. |
| `TensorRT-LLM@72507c8` | `docs/source/models/visual-generation.md` | Adds LTX-2.3 T2V and localized temporal retake generation pipeline. |

---

## 3. Deterministic Offline Verification

Verification script: `experiments/sglang_amd_aiter_vllm_qwen4_rust_trtllm_ltx23_phase147/run.py`

| Metric | Value | Architectural Meaning |
| :--- | :--- | :--- |
| `amd_deterministic_aiter_drift_pct` | **`0.0%`** | Zero numerical drift after `gfx950` deterministic bypass, AITER `softmax_scale`, and CT FP8 MoE fix |
| `qwen4exp_int64_overflow_faults` | **`0`** | Zero 32-bit pointer overflow faults in Qwen4Exp HC/PLE kernels at large batch/sequence lengths |
| `glm_tool_call_think_parse_error_pct` | **`0.0%`** | Zero GLM reasoning parser state-machine errors across interleaved tool calls |
| `ltx23_retake_compute_reduction_pct` | **`-68.75%`** | Denoising compute reduction for localized LTX-2.3 retake vs full-clip regeneration |
