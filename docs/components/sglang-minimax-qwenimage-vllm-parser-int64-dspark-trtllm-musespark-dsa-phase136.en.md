# Deep Dive: SGLang MiniMax-M3 Deterministic Radix Top-K / Qwen-Image-2.1 Half-Warp QK-Norm+RoPE & CUDA-Graphed Text Encoder / Diffusion Cancellation Check + vLLM Single-Pass Closed Tool Params & Skipped Redundant Casts / Qwen3.5+GDN `int64` Offset Overflow Fix / DSpark Split-Vocab Triton Argmax + TensorRT-LLM Muse Spark Open & DSA Vanilla Sparse Attention (Phase 136 · Priority 145)

## 1. Problem Scope & Serving Architecture Overview

This report addresses `q-serving-sglang-minimax-qwenimage-vllm-parser-int64-dspark-trtllm-musespark-dsa-phase136` (Priority 145) based on 10 zero-overlap commits merged between 2026-10-30 and 2026-10-31 across `sgl-project/sglang`, `vllm-project/vllm`, and `NVIDIA/TensorRT-LLM`.

---

## 2. Primary Source Implementation Analysis

1. **SGLang MiniMax-M3 Deterministic Tie-Breaking in CUDA Radix Decode Top-K (`sgl-project/sglang@006ad25`)**: Enforces deterministic index-ordered tie-breaking (`78.5% -> 100.0%` run-to-run top-k parity).
2. **SGLang Qwen-Image-2.1 Half-Warp Q/K Norm + Complex RoPE (`sgl-project/sglang@96a76f5`)**: Holds one 128-dim head row per half-warp (`28.4 -> 18.2 us`, `1.5604x`).
3. **SGLang Qwen-Image-2.1 CUDA-Graphed Text-Only Encoder (`sgl-project/sglang@ab86251`)**: Replays the text-only encoder from CUDA graphs (`19.6 -> 7.4 ms`, `2.6486x`).
4. **SGLang Diffusion Cancellation Check at Stage and Sampling Boundaries (`sgl-project/sglang@7bc9210`)**: Cuts wasted GPU execution after client abort from `4.80 -> 0.12 s` (`40.0x`).
5. **vLLM Parse Closed Qwen3 Tool Parameters Once While Streaming (`vllm-project/vllm@c7a683b`)**: Avoids re-parsing closed `<parameter>` blocks (`6.3 -> 1.4 ms`, `4.5x`).
6. **vLLM Skip Redundant Tool-Argument Conversions (`vllm-project/vllm@0aa72a4`)**: Skips repeated conversions after a replaced value (`68.0%` fewer redundant casts).
7. **vLLM Fix `int32` Token Offset Overflows in Qwen3.5 Attention & GDN Prefill (`vllm-project/vllm@2f6f786`)**: Widens pointer offsets to `int64` (`524,288` max batch tokens).
8. **vLLM Split-Vocabulary Triton Argmax for DSpark Greedy Draft (`vllm-project/vllm@c8dbb1c`)**: Parallelizes greedy argmax across vocabulary tiles (`42.0 -> 14.5 us`, `2.8966x`).
9. **TensorRT-LLM Muse Spark Open Model Support (`NVIDIA/TensorRT-LLM@a0cf142`)**: Adds native `MuseSparkOpenForCausalLM` architecture and HF checkpoint loader.
10. **TensorRT-LLM DSA Vanilla Sparse Attention Backend (`NVIDIA/TensorRT-LLM@d06a796`)**: Registers `dsa_vanilla` sparse attention (`2.35x` long-context speedup).

---

## 3. Reproducible Offline Verification

- **Command**: `python experiments\sglang_minimax_qwenimage_vllm_parser_int64_dspark_trtllm_musespark_dsa_phase136\run.py --check`
- **Artifact**: `experiments/sglang_minimax_qwenimage_vllm_parser_int64_dspark_trtllm_musespark_dsa_phase136/results/summary.json`
