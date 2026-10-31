# Deep Dive: SGLang Shared PD Handoff Token Commit / Deterministic TP>1 Top-P & Top-K Renorm / GLM-5.3 CP+TP Decode Fusion / Async PD Last-Chunk D2H + vLLM Async DP Dummy-Batch Alignment / Qwen3.5 Fused Norm+Quant & Gate+Quant / Direct BF16 Persistent Top-K + TensorRT-LLM OpenEngine Image Routing / Idle Disagg Cancellation / DynamicEPLB Kernels (Phase 134 · Priority 143)

## 1. Problem Scope & Serving Architecture Overview

This report addresses `q-serving-sglang-pd-commit-tp-renorm-glm53-vllm-dp-qwen35-dsa-trtllm-eplb-phase134` (Priority 143) based on 10 zero-overlap commits merged between 2026-10-30 and 2026-10-31 across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`.

---

## 2. Primary Source Implementation Analysis

1. **SGLang Shared PD Handoff Per-Token Commit (`sgl-project/sglang@4df3895`)**: Routes the PD prefill handoff token through the unified decode commit pipeline, preserving first-token `logprob`, `top_logprobs`, and reasoning terminator state.
2. **SGLang Deterministic Top-P / Top-K Renorm When `tp_size > 1` (`sgl-project/sglang@a2cafa5`)**: Enforces `is_deterministic=True` (integer-histogram AIR top-p and fixed-point radix top-k) whenever `tp_size > 1`, eliminating float-atomic tail divergence (`0.18% -> 0.0%`) in DFlash/DSpark speculative verification.
3. **SGLang GLM-5.3 TP Decode Fusion Restoration with CP Prefill (`sgl-project/sglang@a959f31`)**: Restores deferred AllReduce+RMSNorm fusion and graph sizing on the ordinary TP group during decode (`16.8 -> 12.4 ms`, `1.3548x`).
4. **SGLang PD Async Last-Chunk State Page Index D2H (`sgl-project/sglang@3e83f3a`)**: Uses `async_d2h` on the forward stream instead of blocking the scheduler (`1.85 -> 0.12 ms`, `15.42x`).
5. **vLLM Offline DP Collective Alignment under Async Scheduling (`vllm-project/vllm@4c38b65`)**: Tracks `last_step_model_executed` and invokes `execute_dummy_batch()` when a step runs no forward pass (`0` collective hangs).
6. **vLLM Qwen3.5 Fused Norm+Quant & Gate+Quant (`vllm-project/vllm@6c19299`)**: Directly emits `fp8_e4m3fn` from GDN gated RMSNorm and sigmoid gate multiplication on SM89+ (`42.0 -> 24.5 us`, `1.7143x`).
7. **vLLM DSA Persistent Top-K Direct BF16 Read (`vllm-project/vllm@e31cf3d`)**: Templates `PersistentTopKParams<InT>` on `bf16`, eliminating the pre-pass FP32 cast and halving memory read bytes (`38.4 -> 22.0 us`, `1.7455x`).
8. **TensorRT-LLM OpenEngine Image Routing Metadata (`NVIDIA/TensorRT-LLM@c8ee74b`)**: Adds `ImageRoutingMetadata(version=1, token_id_offset)` to `BlockStored` KV events (`41.0% -> 89.0%` multimodal prefix hit rate).
9. **TensorRT-LLM Idle Disaggregated Worker Cancellation (`NVIDIA/TensorRT-LLM@1a1ba9a`)**: Adds `supports_idle_request_cancellation` and rank-synced error voting (`250.0 -> 4.0 ms` cancellation latency, `62.5x`).
10. **TensorRT-LLM DynamicEPLB Kernels & Runtime (`NVIDIA/TensorRT-LLM@b6d1a06`)**: Integrates `moeRebalanceHaloQ.cu` dynamic expert load balancing, reducing max-to-mean expert load ratio `1.85 -> 1.12` (`1.3846x` MoE throughput).

---

## 3. Reproducible Offline Verification

- **Command**: `python experiments\sglang_pd_commit_tp_renorm_glm53_vllm_dp_qwen35_dsa_trtllm_eplb_phase134\run.py --check`
- **Artifact**: `experiments/sglang_pd_commit_tp_renorm_glm53_vllm_dp_qwen35_dsa_trtllm_eplb_phase134/results/summary.json`
