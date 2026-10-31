# Deep Dive: SGLang NPU MiMo-V2 `attention_value_scale` Folding / H20 Qwen3-30B-A3B Block-FP8 Configs / Private-Stream CUDA Graph Pre-Upload / Retracted PD Request Abort / HiCache Forward-Fenced Read-Done + vLLM Incremental Custom-Tool Decoding / Default EPLB Redundant Experts / Qwen3.5 Multi-Layer MTP Fix / RL NCCL Weight Update Cache Release + TensorRT-LLM SM120 DeepSeek-V4.1-Flash (Phase 135 · Priority 144)

## 1. Problem Scope & Serving Architecture Overview

This report addresses `q-serving-sglang-mimo-h20-cudagraph-hicache-vllm-tool-eplb-mtp-nccl-trtllm-sm120-phase135` (Priority 144) based on 10 zero-overlap commits merged between 2026-10-30 and 2026-10-31 across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`.

---

## 2. Primary Source Implementation Analysis

1. **SGLang NPU MiMo-V2 `attention_value_scale` Folding into `o_proj` (`sgl-project/sglang@c729000`)**: Folds `v_scale` into `o_proj` weights at load time (`self.v_scale_folded = True`), eliminating the runtime per-token V multiply (`48.0 -> 41.2 us`, `1.1650x`).
2. **SGLang H20 Block-FP8 (`128x128`) Triton Configs for Qwen3-30B-A3B (`sgl-project/sglang@319b982`)**: Adds tuned Triton MoE tile configs (`E=128, N=768`) for NVIDIA H20 (`96.0 -> 72.5 us`, `1.3241x`).
3. **SGLang Private-Stream Deduplicated CUDA Graph Pre-Upload (`sgl-project/sglang@be556ff`)**: Uploads each deduplicated `GraphExecGroup` on a private `upload_stream` during `seal()` before its first launch, cutting first-replay stall `14.5 -> 1.8 ms` (`8.0556x`).
4. **SGLang PD Scheduler Retracted Request Abort (`sgl-project/sglang@c105a91`)**: Computes `pre_alloc_fill_len(req)` excluding the unmaterialized pending decode token and cleanly aborts retracted requests whose KV cannot be rebuilt.
5. **SGLang HiCache Forward-Fenced Prefill `read-done` Publication (`sgl-project/sglang@9f44526`)**: Publishes `shared_read_done_event` early when KV offload copies are fenced on `forward_stream` (`1.24x` schedule-forward overlap).
6. **vLLM Incremental Streamed Custom-Tool Input Decoding (`vllm-project/vllm@8777b4c`)**: Adds `extend_custom_tool_input_prefix` to decode only `arguments[end:]` per step, turning $O(N^2)$ string re-parsing into $O(N)$ (`18.4 -> 0.92 ms` at 4K tokens, `20.0x`).
7. **vLLM Default EPLB `num_redundant_experts` to Minimum Valid Value (`vllm-project/vllm@8ecf86b`)**: Automatically computes the minimum valid redundant expert count via `compute_eplb_num_redundant_experts`.
8. **vLLM Multi-Layer MTP Module Selection Fix for Qwen3.5 / Qwen3-Next (`vllm-project/vllm@dd5f5ee`)**: Restores multi-module MTP routing and speculative acceptance rate (`62.0% -> 84.5%`).
9. **vLLM RL NCCL Weight Transfer Cache Release (`vllm-project/vllm@be712f5`)**: Calls `torch.accelerator.empty_cache()` in `finish_weight_update()`, freeing `3.2 GiB` of temporary reload buffers for co-located RL/NCCL consumers.
10. **TensorRT-LLM DeepSeek-V4.1-Flash on SM120 (`NVIDIA/TensorRT-LLM@c94fc1c`)**: Fixes `SM120BlockScaledKernel` 1D-1D FP8 GEMM synchronization and `csa2` sparse attention plan reuse on SM120 (`1.42x` throughput).

---

## 3. Reproducible Offline Verification

- **Command**: `python experiments\sglang_mimo_h20_cudagraph_hicache_vllm_tool_eplb_mtp_nccl_trtllm_sm120_phase135\run.py --check`
- **Artifact**: `experiments/sglang_mimo_h20_cudagraph_hicache_vllm_tool_eplb_mtp_nccl_trtllm_sm120_phase135/results/summary.json`
