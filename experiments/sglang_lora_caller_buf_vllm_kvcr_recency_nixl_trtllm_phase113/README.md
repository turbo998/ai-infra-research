# Phase 113 Track B: SGLang Diffusion LoRA Scaling & BCG Caller Buffers, vLLM KVCR Prefix Recency & Rust Grammar/NIXL Telemetry, and TRT-LLM GH200 / DSv3.2

Deterministic Python 3.12 standard-library verification suite for Phase 113 (`priority: 122`) Track B:
- `sgl-project/sglang@dc41126cde`, `a912002ee2`, `6724403444` & `5044779f0f`: AI-Toolkit Flux LoRA `move_alpha` + `.lora_A` suffix matching (`2.0` vs over-scaled `4.0`), caller-allocated breakable CUDA graph output buffers (`0` dynamic allocs), truncated multimodal pad-span clipping (`4` in-bounds tokens), and AMD `(architecture, model_type)` mono-kernel registry
- `vllm-project/vllm@e3883b7a7e`, `a23d7dc3f1`, `c56cdbea15` & `405276b8f0`: KVCR `align_sequence` prefix-recency ordering (`3` vs `0` contiguous prefix hits after eviction), CPU encoder bidirectional attention, Rust NIXL stats decoding (`6` fields), and comment-aware `grammar_is_likely_lark`
- `NVIDIA/TensorRT-LLM@e805aac356` & `c932dd8abb`: GH200 default model storage & DeepSeek-V3.2 tools preamble (`4` restored blank lines)

```powershell
python experiments/sglang_lora_caller_buf_vllm_kvcr_recency_nixl_trtllm_phase113/run.py --check
```
