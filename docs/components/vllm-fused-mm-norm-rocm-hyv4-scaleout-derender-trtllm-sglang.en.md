# Deep Dive into vLLM `FusedMMInputNorm` Non-Persistent Buffers, ROCm HY-V4 Preview Fusion, Scale-Out `output_mode="derender"`, Rust Custom Histogram Buckets & Structured Outputs, TRT-LLM Video Hash Parity & SGLang NPU Init

> **Language / 语言**: [English](vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang.en.md) | [中文](vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang.md)  
> **Phase**: Phase 49 (`priority: 58`) · **Date**: 2026-10-22  
> **Primary Upstream Sources**:
> - `vllm-project/vllm@90c4a1a73eecea0428ee4303ad96c9fdeaac3f80` (`#62767`, 2026-10-21)
> - `vllm-project/vllm@833349105b9581d19487295454c19c924e8a3b37` (`#58509`, 2026-10-21)
> - `vllm-project/vllm@817fa3518cd4bcc94b31e044c534c002e701c801` (`#60082`, 2026-10-21)
> - `vllm-project/vllm@c1210ccbe5eab9ad5ee1213f0fc09c3ff13e4bea` (`#62826`, 2026-10-21)
> - `vllm-project/vllm@18e8791134e8e46134f982c8263d5fa0a50548c0` (`#62823`, 2026-10-21)
> - `NVIDIA/TensorRT-LLM@12accb2d5e888bf7dc003fc83c0993b4eab86a75` (`#19920`, 2026-10-21)
> - `sgl-project/sglang@2815eed49c94dbc3266fb5dc4b01c61fc8eb73ca` (`#43039`, 2026-10-21)

---

## 1. Context & Problem Statement

Across multimodal preprocessing, ROCm HY-V4 execution, disaggregated scale-out serving, the vLLM Rust frontend, and TensorRT-LLM / SGLang hardware runtimes, seven upstream engineering bottlenecks were resolved:

1. **`FusedMMInputNorm` Overwritten by `DummyModelLoader` (`vllm#62767`)**: `FusedMMInputNorm` registered processor-derived `weight = rescale_factor / std` and `bias = -mean / std` with PyTorch's default `persistent=True`. Under `--load-format dummy`, `DummyModelLoader` overwrote all `state_dict` buffers with dummy values, corrupting image normalization.
2. **ROCm HY-V4 Preview Kernel & Signature Bottlenecks (`vllm#58509`)**: HY-V4 lacked a fused post/pre RMSNorm kernel for multi-stream IHC residuals on ROCm, `AiterMxfp8Experts` rejected gated `MoEActivation.SILU`, and `deepgemm_fp8_paged_mqa_logits` needed signature introspection across AITER versions.
3. **Scale-Out `/v1/generate` Missing `output_mode="derender"` (`vllm#60082`)**: Disaggregated generate servers could only emit token IDs or unparsed text, requiring an extra round-trip to `/derender` to parse tool calls and reasoning blocks.
4. **vLLM Rust Frontend Histogram Buckets & `response_format` Precedence (`vllm#62826`, `vllm#62823`)**: The Rust frontend lacked `--custom-histogram-buckets` across the 8 core metric families and previously let `structured_outputs` shadow `response_format`.
5. **TensorRT-LLM Offline/Serve Video Cache Hash Divergence (`TensorRT-LLM#19920`) & SGLang NPU Warmup Ordering (`sglang#43039`)**: TRT-LLM's offline `load_video` bypassed `VideoMediaIO` and its `raw_bytes_hash`, breaking prefix-cache parity with `trtllm-serve`; SGLang called `_resolve_backend` before `init_npu_backend()`, causing `pre_warm_nccl` to fail on Ascend NPUs.

---

## 2. Architectural & Implementation Analysis

### 2.1 Non-Persistent `FusedMMInputNorm` Buffers (`vllm#62767`)
- `FusedMMInputNorm` now registers `self.register_buffer("weight", weight, persistent=False)` and `self.register_buffer("bias", bias, persistent=False)`.
- Because non-persistent buffers are excluded from `state_dict()`, `DummyModelLoader` leaves processor-derived normalization constants untouched (preserving `< 1e-12` error vs `97.4946` max error when overwritten to `0.5`).

### 2.2 ROCm HY-V4 Preview Optimizations (`vllm#58509`)
- **`triton_ihc_post_pre_rms_norm`**: Fuses 5 operations—post-RMSNorm of `x` with `w_post`, scaled residual accumulation `x_h + scale_2 * y_post`, in-place `x_h` update, and pre-RMSNorm of all `n` streams with `w_pre`—into a single Triton launch (`5 -> 1` launch reduction per layer).
- **`AiterMxfp8Experts` Gated SiLU Admission**: Adds `fused_moe_supports_gate_mode()`, admitting `MoEActivation.SILU` and `MoEActivation.GELU` alongside `SILU_NO_MUL` (`gate_mode=False`).
- **MQA Logits Introspection**: Uses `_accepts_parameters(rocm_aiter_ops.deepgemm_fp8_paged_mqa_logits, ["chunked_Size", "cu_seqlens_k"])` to guard local-attention arguments on `gfx942/gfx950`.

### 2.3 Scale-Out `output_mode="derender"` & `ParseContext` (`vllm#60082`)
- `/render` populates `ParseContext` (`tools`, `tool_choice`, `parallel_tool_calls`, `include_reasoning`, JSON-validated `chat_template_kwargs`, and `history_tool_call_cnt`).
- Intermediate scale-out stages (`output_mode="tokens"`, e.g. disaggregated prefill) accept `ParseContext` with unknown fields (`extra="allow"`), whereas `output_mode="derender"` strictly rejects unknown fields or `stream=True` and invokes `OnlineDerenderer._derender_text_choices` to return parsed `ChatMessage` choices directly from `/v1/generate`.

### 2.4 Rust Frontend, TensorRT-LLM Video Hash Parity & SGLang NPU Init (`#62826`, `#62823`, `#19920`, `#43039`)
- **Rust `--custom-histogram-buckets` (`vllm#62826`)**: Supports replacing bucket boundaries across 8 `BucketFamily` keys (`request_latency`, `time_to_first_token`, `inter_token_latency`, `iteration_tokens`, `request_params_n`, `request_num_preemptions`, `request_tokens`, `kv_cache_residency`), enforcing non-empty, finite, positive, strictly increasing bounds.
- **Rust `convert_from_response_format` (`vllm#62823`)**: Overrides same-discriminant constraints in `structured_outputs` while rejecting conflicting constraint kinds with `400 Invalid Request`.
- **TRT-LLM Video Hash Parity (`TensorRT-LLM#19920`) & SGLang NPU Init (`sglang#43039`)**: TRT-LLM routes offline `load_video` (`path`, `file://`, `data:video/mp4;base64`, `http(s)://`) through `VideoMediaIO` with a shared `200 MiB` (`_MAX_RESPONSE_BYTES`) cap and identical `raw_bytes_hash`; SGLang invokes `init_npu_backend()` at the start of `init_parallel_runtime` before `_resolve_backend`.

---

## 3. Deterministic Standard-Library Verification

Verified via `experiments/vllm-fused-mm-norm-rocm-hyv4-scaleout-derender-trtllm-sglang/run.py --check`:

| Dimension | Invariant / Metric | Reproduced Value | Status |
|---|---|---|---|
| `FusedMMInputNorm` Fidelity | `persistent=False` vs Overwritten `0.5` Max Error | `< 1e-12` vs `97.4946` | `PASS` |
| ROCm HY-V4 IHC Norm Fusion | Kernel Launches per Layer | `5 -> 1` (**`5.0×`** launch reduction) | `PASS` |
| Scale-Out `ParseContext` Validation | Prefill (`tokens`) Ignores Unknowns / `derender` Rejects | `ok:tokens` / `error:unknown_parse_context_fields:tool_choise` | `PASS` |
| Rust Frontend Buckets & Precedence | 8 Bucket Families + Same-Kind Override | `8` families validated, `json_schema:schema_v2` | `PASS` |
| TRT-LLM Video Cache Identity | 4-Source `raw_bytes_hash` Parity & `200 MiB` Cap | `trtllm_video_hash_parity = True` (`209,715,200 B` cap) | `PASS` |
